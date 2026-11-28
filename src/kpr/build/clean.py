"""Clean interim tables and run automatic quality checks.

Owner: data session. Task D10 in PARALLEL_SESSIONS.md §7. Dossier §6.4, §6.7.

`python -m kpr.build.clean` cleans in place, then checks, then writes a report to
raw/quality/quality_report.md, and exits with code 1 if a rule fails ("échoue bruyamment").

Cleaning (dossier §6.4):
- prices: a daily simple return outside [-80 %, +200 %] is impossible for these stocks and is
  set to missing (adjustment error); moves beyond ±40 % are listed for a manual look, not
  changed.
- wiki_views: a robot spike is a day whose log views jump far above the person's recent
  level (robust z-score > `spike_z`) while neither GDELT (article count) nor the
  local-language page moves (z < `confirm_z`). Such days are set to missing. The rule needs
  GDELT: persons without gdelt_daily rows are left untouched. Company pages are not cleaned.
  (GDELT deduplication and the 3-article minimum for tone are done at collection, kpr.collect.
  gdelt; the relative tone is computed with the panel, task D11.)

Checks (dossier §6.7), each pass / warn / fail / skip:
1. every interim table present matches its frozen schema (types, no duplicate keys);
2. views series: at most 5 % missing days during the person's mandates, else flagged (warn);
3. every person with a Wikipedia page has views; every person has GDELT articles;
4. no overlapping CEO, CFO or chair mandates in one company, except declared co-CEOs;
5. daily returns within [-80 %, +200 %];
plus a warning for company-years with an implausible number of earnings releases.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from kpr.collect.common import load_params, load_persons
from kpr.schemas import SchemaError, validate
from kpr.utils.io import raw_dir, table_path, write_table
from kpr.utils.logging import get_logger

log = get_logger(__name__)

SINGLE_ROLES = ("CEO", "CFO", "chair")  # one holder at a time; "other" and "founder" can overlap
DATA_TABLES = ("prices", "market", "vix", "wiki_views", "gdelt_daily", "trends_ratio",
               "earnings", "mandates", "headlines", "panel")


@dataclass
class Check:
    name: str
    status: str  # pass | warn | fail | skip
    detail: str = ""


def read_if_exists(name: str) -> pd.DataFrame | None:
    path = table_path(name)
    return pd.read_parquet(path) if path.exists() else None


# ---- cleaning -------------------------------------------------------------------------------

def clean_prices(prices: pd.DataFrame, ret_min: float, ret_max: float,
                 flag_abs: float = 0.4) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (cleaned prices, impossible rows set to missing, large moves to look at)."""
    simple = np.expm1(prices["ret"])
    impossible = (simple < ret_min) | (simple > ret_max)
    large = (simple.abs() > flag_abs) & ~impossible
    out = prices.copy()
    out.loc[impossible, "ret"] = np.nan
    cols = ["company_id", "date", "ret"]
    return out, prices.loc[impossible, cols], prices.loc[large, cols]


SCALE_FLOOR = 0.25  # log units: a flat series (e.g. 0 articles for weeks) still has a scale


def robust_z(x: pd.Series, window: int) -> pd.Series:
    """(x - rolling median) / max(1.4826 * rolling MAD, SCALE_FLOOR), previous `window` days."""
    past = x.shift(1).rolling(window, min_periods=window // 2)
    med = past.median()
    mad = (x.shift(1) - med).abs().rolling(window, min_periods=window // 2).median()
    return (x - med) / (1.4826 * mad).clip(lower=SCALE_FLOOR)


def bot_spikes(views: pd.DataFrame, articles: pd.Series | None, spike_z: float,
               confirm_z: float, window: int) -> pd.Series:
    """Boolean mask (index = date) of robot spikes for one person's views."""
    v = views.set_index("date")
    z_total = robust_z(np.log1p(v["views_total"]), window)
    spike = z_total > spike_z
    if articles is None or not spike.any():
        return pd.Series(False, index=v.index)
    a = articles.reindex(v.index)
    z_news = robust_z(np.log1p(a), window)
    news_moves = z_news > confirm_z
    has_local = v["views_local"].fillna(0).gt(0).any()
    if has_local:
        local_moves = robust_z(np.log1p(v["views_local"]), window) > confirm_z
    else:
        local_moves = pd.Series(False, index=v.index)
    return (spike & ~news_moves.fillna(False) & ~local_moves.fillna(False)).fillna(False)


def clean_wiki(wiki: pd.DataFrame, gdelt: pd.DataFrame | None, spike_z: float,
               confirm_z: float, window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Set robot spikes to missing; return (cleaned table, removed rows)."""
    out = wiki.copy()
    removed = []
    if gdelt is None or gdelt.empty:
        return out, pd.DataFrame(columns=["page_id", "date", "views_total"])
    by_person = {pid: g.set_index("date")["n_articles"] for pid, g in gdelt.groupby("person_id")}
    for page_id, g in wiki.groupby("page_id"):
        if page_id not in by_person:
            continue  # company pages, or persons without GDELT rows
        mask = bot_spikes(g, by_person[page_id], spike_z, confirm_z, window)
        if mask.any():
            idx = g.index[mask.values]
            removed.append(wiki.loc[idx, ["page_id", "date", "views_total"]])
            out.loc[idx, ["views_en", "views_local", "views_total"]] = np.nan
    gone = pd.concat(removed) if removed else pd.DataFrame(
        columns=["page_id", "date", "views_total"])
    return out, gone


# ---- checks ---------------------------------------------------------------------------------

def check_schemas(tables: dict[str, pd.DataFrame | None]) -> list[Check]:
    out = []
    for name, df in tables.items():
        if df is None:
            out.append(Check(f"schema {name}", "skip", "table not built yet"))
            continue
        try:
            validate(df, name)
            out.append(Check(f"schema {name}", "pass", f"{len(df)} rows"))
        except SchemaError as exc:
            out.append(Check(f"schema {name}", "fail", str(exc)))
    return out


def mandate_days(persons: list[dict], start: str, end: str) -> dict[str, pd.DatetimeIndex]:
    """Days in office within the study period, per person (all mandates together)."""
    days: dict[str, pd.DatetimeIndex] = {}
    for p in persons:
        s = max(pd.Timestamp(str(p["start"])), pd.Timestamp(start))
        e = min(pd.Timestamp(str(p["end"])) if p.get("end") else pd.Timestamp(end),
                pd.Timestamp(end))
        if s <= e:
            rng = pd.date_range(s, e, freq="D")
            days[p["person_id"]] = days.get(p["person_id"], pd.DatetimeIndex([])).union(rng)
    return days


def check_views(wiki: pd.DataFrame | None, persons: list[dict], max_missing: float,
                start: str, end: str) -> list[Check]:
    if wiki is None:
        return [Check("views coverage", "skip", "wiki_views not built yet")]
    with_page = {p["person_id"] for p in persons if (p.get("wiki_pages") or {})}
    all_ids = {p["person_id"] for p in persons}
    pages = set(wiki["page_id"])
    no_data = sorted(with_page - pages)
    out = [Check("every person with a page has views", "fail" if no_data else "pass",
                 ", ".join(no_data))]
    no_page = sorted(all_ids - with_page)
    if no_page:
        out.append(Check("persons without any Wikipedia page", "warn",
                         f"{len(no_page)} (fail the 200 views filter of §4.4): "
                         + ", ".join(no_page)))
    api_start = max(pd.Timestamp(start), wiki["date"].min())
    days = mandate_days(persons, str(api_start.date()), end)
    flagged = []
    for pid, g in wiki[wiki["page_id"].isin(days)].groupby("page_id"):
        s = g.set_index("date")["views_total"].reindex(days[pid])
        share = s.isna().mean()
        if share > max_missing:
            flagged.append(f"{pid} {share:.0%}")
    out.append(Check(f"views missing <= {max_missing:.0%} during mandates",
                     "warn" if flagged else "pass", ", ".join(flagged)))
    return out


def check_articles(gdelt: pd.DataFrame | None, persons: list[dict]) -> Check:
    if gdelt is None:
        return Check("every person has GDELT articles", "skip", "gdelt_daily not built yet")
    totals = gdelt.groupby("person_id")["n_articles"].sum()
    missing = sorted({p["person_id"] for p in persons} - set(totals[totals > 0].index))
    return Check("every person has GDELT articles", "fail" if missing else "pass",
                 ", ".join(missing))


def check_mandates(persons: list[dict]) -> Check:
    clashes = []
    df = pd.DataFrame(persons)
    df["start"] = pd.to_datetime(df["start"].astype(str))
    df["end"] = pd.to_datetime(df["end"].map(lambda v: str(v) if v else None),
                               format="%Y-%m-%d", errors="coerce")
    co = df["co_ceo"].map(bool) if "co_ceo" in df else pd.Series(False, index=df.index)
    single = df["role"].isin(SINGLE_ROLES) & ~co
    for (cid, role), g in df[single].groupby(["company_id", "role"]):
        g = g.sort_values("start")
        prev_end, prev_id = None, None
        for r in g.itertuples():
            if prev_id is not None and (pd.isna(prev_end) or r.start <= prev_end):
                clashes.append(f"{cid} {role}: {prev_id} / {r.person_id}")
            if prev_id is None or pd.isna(r.end) or (not pd.isna(prev_end) and r.end > prev_end):
                prev_end, prev_id = r.end, r.person_id
    return Check("no overlapping mandates (except co-CEOs)", "fail" if clashes else "pass",
                 "; ".join(clashes))


def check_returns(prices: pd.DataFrame | None, ret_min: float, ret_max: float) -> Check:
    if prices is None:
        return Check("returns within bounds", "skip", "prices not built yet")
    simple = np.expm1(prices["ret"].dropna())
    bad = int(((simple < ret_min) | (simple > ret_max)).sum())
    return Check(f"returns within [{ret_min:.0%}, {ret_max:+.0%}]", "fail" if bad else "pass",
                 f"{bad} rows" if bad else "")


def check_earnings(earnings: pd.DataFrame | None) -> Check:
    if earnings is None:
        return Check("earnings releases per year", "skip", "earnings not built yet")
    from kpr.collect.common import load_universe
    from kpr.collect.earnings import incomplete_years

    odd = incomplete_years(earnings, load_universe(), load_params()["period"]["end"])
    detail = ", ".join(f"{r.company_id} {r.year}: {r.releases}" for r in odd.itertuples())
    return Check("earnings releases per year plausible", "warn" if len(odd) else "pass", detail)


def report(checks: list[Check], removed_spikes: pd.DataFrame, impossible: pd.DataFrame,
           large: pd.DataFrame) -> str:
    lines = ["# Data quality report", "", f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}", "",
             "| check | status | detail |", "| --- | --- | --- |"]
    for c in checks:
        lines.append(f"| {c.name} | {c.status.upper()} | {c.detail[:600]} |")
    lines += ["", f"## Robot spikes set to missing: {len(removed_spikes)}", ""]
    lines += [f"- {r.page_id} {r.date:%Y-%m-%d} ({r.views_total:.0f} views)"
              for r in removed_spikes.itertuples()][:200]
    lines += ["", f"## Impossible returns set to missing: {len(impossible)}", ""]
    lines += [f"- {r.company_id} {r.date:%Y-%m-%d} log ret {r.ret:.3f}"
              for r in impossible.itertuples()]
    lines += ["", f"## Daily moves beyond ±40 % kept, to look at: {len(large)}", ""]
    lines += [f"- {r.company_id} {r.date:%Y-%m-%d} {np.expm1(r.ret):+.0%}"
              for r in large.itertuples()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kpr.build.clean")
    parser.add_argument("--check-only", action="store_true", help="do not modify tables")
    args = parser.parse_args(argv)

    params = load_params()
    q, c = params["quality"], params.get("cleaning", {})
    start, end = params["period"]["start"], params["period"]["end"]
    persons = load_persons()

    prices, wiki, gdelt = (read_if_exists(n) for n in ("prices", "wiki_views", "gdelt_daily"))
    impossible = large = removed = pd.DataFrame()
    if prices is not None:
        prices, impossible, large = clean_prices(prices, q["ret_min"], q["ret_max"],
                                                 c.get("flag_abs_return", 0.4))
        if not args.check_only and len(impossible):
            write_table(prices, "prices")
    if wiki is not None:
        wiki, removed = clean_wiki(wiki, gdelt, c.get("spike_z", 5.0), c.get("confirm_z", 2.0),
                                   c.get("spike_window", 60))
        if not args.check_only and len(removed):
            write_table(wiki, "wiki_views")

    tables = {n: read_if_exists(n) for n in DATA_TABLES}
    checks = check_schemas(tables)
    checks += check_views(tables["wiki_views"], persons, q["max_missing_share_views"], start, end)
    checks.append(check_articles(tables["gdelt_daily"], persons))
    checks.append(check_mandates(persons))
    checks.append(check_returns(tables["prices"], q["ret_min"], q["ret_max"]))
    checks.append(check_earnings(tables["earnings"]))

    text = report(checks, removed if len(removed) else pd.DataFrame(
        columns=["page_id", "date", "views_total"]), impossible, large)
    path = raw_dir("quality") / "quality_report.md"
    path.write_text(text, encoding="utf-8")
    for ch in checks:
        level = {"fail": 40, "warn": 30}.get(ch.status, 20)
        log.log(level, "%-5s %s %s", ch.status.upper(), ch.name, ch.detail[:200])
    failed = [ch for ch in checks if ch.status == "fail"]
    log.info("report: %s", path)
    if failed:
        log.error("%d quality checks failed", len(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
