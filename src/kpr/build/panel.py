"""Build the daily (person, trading day) panel.

Owner: data session. Task D11 in PARALLEL_SESSIONS.md §7. Dossier §6.6, §6.5, §7.1, §6.4, §4.4.

One row per person and trading session of the exchange of the company they lead, during
their mandates within the study period. Columns follow the frozen `panel` contract.

- Attention z-scores (dossier §7.1) are computed on the daily UTC series:
  z_t = (log(1 + x_t) - mean(log(1 + x) over [t-65, t-6])) / std(same window), for views
  (z_wiki) and for GDELT article counts (z_news). At least `min_obs` of the 60 days must be
  present.
- tone_rel (dossier §6.4): the day's GDELT mean tone minus its mean over the previous 90 days,
  divided by the standard deviation over the same days (tone is missing under 3 articles).
- Alignment (dossier §6.5): a UTC day d belongs to session J0 = d if d is a trading day of the
  exchange, else the next session. When several UTC days fall on one session (a weekend
  before Monday), the panel keeps the day with the largest z_wiki, with all its attention
  columns, so that a Saturday shock shows up on Monday instead of being lost.
- Sessions come from kpr.utils.calendar (task Q1) when it is implemented, else from the dates
  of the company's own price series (same sessions, minus rare data gaps).
- ret: the company's log return on the session; vix: VIX close of the session (last known
  value on US holidays); earnings_window: an earnings release within ±k sessions, k =
  filters.earnings_window_sessions of config/params/quant.yaml.
- Selection (dossier §4.4 steps 2 and 3): persons whose median daily views during their
  mandates are under `min_median_daily_views`, or whose median Trends ratio is under
  `min_trends_ratio` (when known), are left out. The decision for every person is saved in
  raw/quality/selection.csv. Step 4 (8 usable shocks) is downstream, after event detection.

Usage: python -m kpr.build.panel [--allow-missing-gdelt] [--no-selection]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml

from kpr.collect.common import CONFIG_DIR, load_params, load_persons, load_universe
from kpr.utils.io import raw_dir, read_table, write_table
from kpr.utils.logging import get_logger

log = get_logger(__name__)

PANEL_COLUMNS = ["person_id", "date", "company_id", "views", "z_wiki", "n_articles", "z_news",
                 "tone_rel", "ret", "earnings_window", "vix"]


def attention_z(x: pd.Series, window: int = 60, gap: int = 5, min_obs: int = 30) -> pd.Series:
    """Dossier §7.1 on a daily series indexed by calendar day (missing days stay missing)."""
    full = x.asfreq("D") if isinstance(x.index, pd.DatetimeIndex) and len(x) else x
    lx = np.log1p(full.astype(float))
    ref = lx.shift(gap + 1).rolling(window, min_periods=min_obs)
    z = (lx - ref.mean()) / ref.std()
    return z.reindex(x.index)


def relative_tone(tone: pd.Series, window: int = 90, min_obs: int = 10) -> pd.Series:
    """Dossier §6.4: tone minus its mean over the previous `window` days, over their std."""
    full = tone.asfreq("D") if len(tone) else tone
    ref = full.shift(1).rolling(window, min_periods=min_obs)
    rel = (full - ref.mean()) / ref.std()
    return rel.reindex(tone.index)


def sessions_for(exchange: str, start: str, end: str,
                 fallback: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Trading sessions from kpr.utils.calendar (Q1) if available, else `fallback`."""
    try:
        from kpr.utils.calendar import trading_days

        days = pd.DatetimeIndex(trading_days(exchange, start, end))
        if len(days):
            return days.normalize()
    except NotImplementedError:
        pass
    return fallback[(fallback >= start) & (fallback <= end)]


def map_to_sessions(days: pd.DatetimeIndex, sessions: pd.DatetimeIndex) -> pd.Series:
    """J0 of every UTC day: the day itself if it is a session, else the next session."""
    pos = sessions.searchsorted(days, side="left")
    j0 = pd.Series(pd.NaT, index=days, dtype="datetime64[ns]")
    ok = pos < len(sessions)
    j0[ok] = sessions[pos[ok]]
    return j0


def person_daily(views: pd.Series, gdelt: pd.DataFrame | None, start: str,
                 end: str) -> pd.DataFrame:
    """Daily UTC attention of one person with z-scores and relative tone."""
    days = pd.date_range(start, end, freq="D")
    v = views.reindex(days)
    df = pd.DataFrame({"views": v, "z_wiki": attention_z(v)}, index=days)
    if gdelt is not None and len(gdelt):
        g = gdelt.set_index("date").reindex(days)
        n = g["n_articles"]
        df["n_articles"] = n
        df["z_news"] = attention_z(n)
        df["tone_rel"] = relative_tone(g["tone_mean"])
    else:
        df["n_articles"] = np.nan
        df["z_news"] = np.nan
        df["tone_rel"] = np.nan
    return df


def collapse_to_sessions(daily: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per session: the UTC day of that session's group with the largest z_wiki."""
    d = daily.copy()
    d["session"] = map_to_sessions(d.index, sessions).values
    d = d[d["session"].notna()]
    d["_rank"] = d["z_wiki"].fillna(-np.inf)
    d["_utc"] = d.index
    # Ties (e.g. all z missing): keep the session day itself, which is the last of the group.
    d = d.sort_values(["session", "_rank", "_utc"])
    best = d.groupby("session").tail(1).set_index("session")
    return best.drop(columns=["_rank", "_utc"]).reindex(sessions)


def earnings_flags(sessions: pd.DatetimeIndex, release_days: pd.Series, k: int) -> np.ndarray:
    """True for sessions within ±k sessions of an earnings release (release mapped to J0)."""
    flags = np.zeros(len(sessions), dtype=bool)
    if len(release_days) == 0 or len(sessions) == 0:
        return flags
    pos = sessions.searchsorted(pd.DatetimeIndex(release_days), side="left")
    for p in pos:
        if p < len(sessions):
            flags[max(0, p - k): p + k + 1] = True
    return flags


def mandate_spans(persons: list[dict], start: str, end: str) -> pd.DataFrame:
    rows = []
    for p in persons:
        s = max(pd.Timestamp(str(p["start"])), pd.Timestamp(start))
        e = min(pd.Timestamp(str(p["end"])) if p.get("end") else pd.Timestamp(end),
                pd.Timestamp(end))
        if s <= e:
            rows.append({"person_id": p["person_id"], "company_id": p["company_id"],
                         "start": s, "end": e})
    return pd.DataFrame(rows, columns=["person_id", "company_id", "start", "end"])


def select_persons(spans: pd.DataFrame, wiki: pd.DataFrame, trends: pd.DataFrame | None,
                   min_views: float, min_ratio: float) -> pd.DataFrame:
    """Dossier §4.4 steps 2 and 3, one row per person with the decision and the reason."""
    rows = []
    views = {pid: g.set_index("date")["views_total"] for pid, g in wiki.groupby("page_id")}
    for pid, g in spans.groupby("person_id"):
        days = pd.DatetimeIndex([])
        for r in g.itertuples():
            days = days.union(pd.date_range(r.start, r.end, freq="D"))
        v = views.get(pid)
        med = float(v.reindex(days).median()) if v is not None else float("nan")
        ratio = float("nan")
        if trends is not None:
            t = trends[(trends.person_id == pid) & trends.year.isin(sorted(set(days.year)))]
            if len(t):
                ratio = float(t["ratio_person_company"].median())
        reasons = []
        if not med >= min_views:
            reasons.append(f"median views {med:.0f} < {min_views:.0f}")
        if not np.isnan(ratio) and ratio < min_ratio:
            reasons.append(f"Trends ratio {ratio:.2f} < {min_ratio:.2f}")
        rows.append({"person_id": pid, "median_views": med, "trends_ratio": ratio,
                     "selected": not reasons, "reason": "; ".join(reasons)})
    return pd.DataFrame(rows)


def build_panel(spans: pd.DataFrame, wiki: pd.DataFrame, gdelt: pd.DataFrame | None,
                prices: pd.DataFrame, vix: pd.DataFrame, earnings: pd.DataFrame,
                exchanges: dict[str, str], start: str, end: str, k: int) -> pd.DataFrame:
    views = {pid: g.set_index("date")["views_total"] for pid, g in wiki.groupby("page_id")}
    news = {pid: g for pid, g in gdelt.groupby("person_id")} if gdelt is not None else {}
    vix_s = vix.set_index("date")["vix"].sort_index()
    frames = []
    for pid, g in spans.groupby("person_id"):
        daily = person_daily(views.get(pid, pd.Series(dtype=float)), news.get(pid), start, end)
        for r in g.itertuples():
            px = prices[prices.company_id == r.company_id].set_index("date")
            sess = sessions_for(exchanges[r.company_id], start, end,
                                pd.DatetimeIndex(px.index).sort_values())
            panel = collapse_to_sessions(daily, sess)
            panel = panel[(panel.index >= r.start) & (panel.index <= r.end)]
            rel = earnings.loc[earnings.company_id == r.company_id, "date"]
            frames.append(pd.DataFrame({
                "person_id": pid,
                "date": panel.index,
                "company_id": r.company_id,
                "views": panel["views"].values,
                "z_wiki": panel["z_wiki"].values,
                "n_articles": panel["n_articles"].fillna(0).astype("int64").values,
                "z_news": panel["z_news"].values,
                "tone_rel": panel["tone_rel"].values,
                "ret": px["ret"].reindex(panel.index).values,
                "earnings_window": earnings_flags(sess, rel, k)[
                    sess.isin(panel.index)],
                "vix": vix_s.reindex(vix_s.index.union(panel.index)).ffill()
                       .reindex(panel.index).values,
            }))
    out = pd.concat(frames, ignore_index=True)
    # Two mandates of one person on the same day (role change): keep the first company.
    out = out.drop_duplicates(["person_id", "date"]).sort_values(["person_id", "date"])
    return out[PANEL_COLUMNS].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kpr.build.panel")
    parser.add_argument("--allow-missing-gdelt", action="store_true",
                        help="build without gdelt_daily (n_articles = 0, no z_news / tone)")
    parser.add_argument("--no-selection", action="store_true",
                        help="keep every person (skip the §4.4 filters)")
    args = parser.parse_args(argv)

    params = load_params()
    start, end = params["period"]["start"], params["period"]["end"]
    sel = params["selection"]
    quant = yaml.safe_load((CONFIG_DIR / "params" / "quant.yaml").read_text(encoding="utf-8"))
    k = int(quant["filters"]["earnings_window_sessions"])

    try:
        gdelt = read_table("gdelt_daily")
    except FileNotFoundError:
        if not args.allow_missing_gdelt:
            log.error("gdelt_daily is missing (task D8). Use --allow-missing-gdelt to build a "
                      "provisional panel without news columns.")
            return 1
        log.warning("building WITHOUT gdelt_daily: n_articles = 0, z_news and tone_rel missing")
        gdelt = None
    wiki = read_table("wiki_views")
    prices, vix, earnings = read_table("prices"), read_table("vix"), read_table("earnings")
    try:
        trends = read_table("trends_ratio")
    except FileNotFoundError:
        trends = None

    spans = mandate_spans(load_persons(), start, end)
    selection = select_persons(spans, wiki, trends, sel["min_median_daily_views"],
                               sel["min_trends_ratio"])
    selection.to_csv(raw_dir("quality") / "selection.csv", index=False)
    if not args.no_selection:
        spans = spans[spans.person_id.isin(selection.loc[selection.selected, "person_id"])]
    log.info("selection: %d of %d persons kept", int(selection.selected.sum()), len(selection))

    exchanges = {c["company_id"]: c["exchange"] for c in load_universe()}
    panel = build_panel(spans, wiki, gdelt, prices, vix, earnings, exchanges, start, end, k)
    write_table(panel, "panel")
    log.info("panel: %d rows, %d persons, %d companies; z_wiki > 3 on %d person-days",
             len(panel), panel.person_id.nunique(), panel.company_id.nunique(),
             int((panel.z_wiki > 3).sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
