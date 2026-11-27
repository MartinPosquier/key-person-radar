"""Attention z-scores, shock detection, clustering, filters, events table.

Owner: quant session. Task Q4, Q5 in PARALLEL_SESSIONS.md §7. Dossier §7.1-7.4, §7.7.

Pipeline (``build_events``):

1. Detection (§7.2) on the daily panel: z_wiki > 3, z_news > 2 and at least 3 articles.
   Successive peak days of a person are one event, dated on the first day: a peak day opens
   a new event only if the person had no peak day in the 10 previous days.
2. Sign (§7.3) from the relative tone of the day: below -1 negative, above +1 positive.
3. J0 (§6.5): first session of the company's exchange on or after the UTC shock day.
4. CARs (§7.5) from ``kpr.build.returns``: [0, 2] target, [-5, -1] anticipation, [0, 1] and
   [0, 5] robustness.
5. Filters (§7.4). Excluded shocks are kept and flagged, with every reason that applies,
   separated by ";": ``earnings`` (within 1 session of a quarterly release),
   ``company_dominant`` (company page z-score above the person's by more than 2),
   ``extreme_market`` (index moves more than 4 % on a day of the [0, 2] window),
   ``short_history`` (fewer than 120 estimation sessions), ``missing_returns`` (no price or
   index data to compute the target CAR).

``finalize`` (Q5) flags natural experiments and their 10-day neighbours as reserved
(§11.2), assigns the split with its 10-session boundary gap (§9.5, exclusion reason
``split_gap``, see ``kpr.build.splits``) and validates the frozen `events` schema (§7.7).
``python -m kpr.build.events`` writes the shared table (real data only; ``--out`` for
fixtures and robustness runs with ``--set key=value``).

The attention z-score of §7.1 compares log(1 + views) with the 60 days ending 6 days before
(a 5-day gap): ``attention_z``. The panel already carries z_wiki and z_news (data session);
``attention_z`` is used for company pages and for robustness runs.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kpr.build import returns as R
from kpr.build import splits
from kpr.schemas import TABLES, validate
from kpr.utils import calendar as cal
from kpr.utils.io import read_table, use_fixtures, write_table
from kpr.utils.logging import get_logger
from kpr.validation import natural_experiments as ne

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXCLUSION_REASONS = ("missing_returns", "short_history", "earnings", "company_dominant",
                     "extreme_market")
CAR_WINDOWS: dict[str, tuple[int, int]] = dict(R.DEFAULT_WINDOWS)
# Minimum number of reference days for a z-score (out of the 60-day window).
Z_MIN_PERIODS = 20


# ---- parameters ------------------------------------------------------------------------------

def _deep_update(base: dict, overrides: Mapping) -> dict:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_params(overrides: Mapping | None = None) -> dict:
    """config/params/quant.yaml, with optional nested overrides (robustness runs, tests)."""
    with open(REPO_ROOT / "config" / "params" / "quant.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return _deep_update(copy.deepcopy(params), overrides or {})


# ---- attention z-score and detection ----------------------------------------------------------

def attention_z(values: pd.Series, window: int = 60, gap: int = 5,
                min_periods: int = Z_MIN_PERIODS, log_scale: bool = True) -> pd.Series:
    """z-score of each day against the `window` days ending `gap` + 1 days before (§7.1).

    `values` is one daily series sorted by date. With window=60 and gap=5 the reference of
    day t is t-65 .. t-6. Missing days stay missing; a flat reference gives NaN.
    """
    x = values.astype(float)
    if log_scale:
        x = np.log1p(x)
    ref = x.shift(gap + 1).rolling(window, min_periods=min_periods)
    sd = ref.std()
    return (x - ref.mean()) / sd.where(sd > 0)


def first_of_clusters(dates: pd.Series, cluster_days: int) -> pd.Series:
    """True for a peak day with no other peak day in the `cluster_days` days before it.

    `dates` must be sorted. Chains of peak days less than `cluster_days` apart form one
    cluster, dated on its first day (§7.2, condition 4).
    """
    gaps = pd.to_datetime(dates).diff().dt.days
    return (gaps.isna() | (gaps > cluster_days)).rename("first")


def detect_shocks(panel: pd.DataFrame, params: Mapping | None = None) -> pd.DataFrame:
    """Shock days of §7.2 from the daily panel, one row per event (first day of a cluster).

    Returns person_id, company_id, date_utc, z_wiki, z_news, n_articles, tone_rel and
    n_peak_days (number of peak days merged into the event).
    """
    sh = (params or load_params())["shocks"]
    peak = panel[
        (panel["z_wiki"] > sh["z_wiki_min"])
        & (panel["z_news"] > sh["z_news_min"])
        & (panel["n_articles"] >= sh["min_articles"])
    ].sort_values(["person_id", "date"])
    cols = ["person_id", "company_id", "date_utc", "z_wiki", "z_news", "n_articles",
            "tone_rel", "n_peak_days"]
    if peak.empty:
        return pd.DataFrame(columns=cols)
    first = peak.groupby("person_id", group_keys=False)["date"].apply(
        lambda d: first_of_clusters(d, sh["cluster_days"])).reindex(peak.index)
    peak = peak.assign(cluster=first.astype(int).groupby(peak["person_id"]).cumsum())
    sizes = peak.groupby(["person_id", "cluster"]).size().rename("n_peak_days")
    events = peak[first.to_numpy()].join(sizes, on=["person_id", "cluster"])
    events = events.rename(columns={"date": "date_utc"})
    return events[cols].reset_index(drop=True)


def sign_of(tone_rel: pd.Series, threshold: float = 1.0) -> pd.Series:
    """negative / neutral / positive from the relative tone (§7.3); missing tone is neutral."""
    t = tone_rel.astype(float)
    out = np.select([t < -threshold, t > threshold], ["negative", "positive"], "neutral")
    return pd.Series(out, index=tone_rel.index, name="sign")


# ---- companies: exchange sessions and market index ---------------------------------------------

@dataclass(frozen=True)
class Company:
    company_id: str
    exchange: str | None = None
    market_index: str | None = None


def load_universe(path: Path | None = None) -> dict[str, Company]:
    """company_id -> Company from config/universe.yaml (empty until task D1 is merged)."""
    path = path or REPO_ROOT / "config" / "universe.yaml"
    with open(path, encoding="utf-8") as f:
        universe = yaml.safe_load(f) or {}
    return {
        c["company_id"]: Company(c["company_id"], c.get("exchange"), c.get("market_index"))
        for c in universe.get("companies") or []
    }


def company_page_id(company_id: str) -> str:
    """page_id of a company's Wikipedia page in the wiki_views table."""
    return f"company_{company_id}"


class Markets:
    """Sessions, J0 and market index of each company.

    Companies listed in universe.yaml use their exchange calendar and index. Others (the
    synthetic fixtures) use the dates of their price rows as sessions and the index whose
    returns correlate best with theirs, with a warning.
    """

    def __init__(self, prices: pd.DataFrame, market: pd.DataFrame,
                 universe: Mapping[str, Company] | None = None):
        self.prices = prices
        self.market = market
        self.universe = dict(load_universe() if universe is None else universe)
        self._index_ret = {i: g.set_index("date")["ret"].sort_index()
                           for i, g in market.groupby("index_id")}
        self._guessed: dict[str, str] = {}
        self.sessions = cache(self._sessions)

    def _sessions(self, company_id: str) -> pd.DatetimeIndex:
        info = self.universe.get(company_id)
        if info and info.exchange:
            last = cal.get_calendar(info.exchange).last_session
            return cal.trading_days(info.exchange, cal.CALENDAR_START, last)
        dates = self.prices.loc[self.prices["company_id"] == company_id, "date"]
        return pd.DatetimeIndex(sorted(dates.unique()), name="date")

    def index_of(self, company_id: str) -> str | None:
        info = self.universe.get(company_id)
        if info and info.market_index:
            return info.market_index
        if company_id not in self._guessed:
            ret = self.prices.loc[self.prices["company_id"] == company_id].set_index("date")["ret"]
            if ret.empty or not self._index_ret:
                return None
            wide = pd.DataFrame(self._index_ret)
            guess = str(wide.corrwith(ret).idxmax())
            log.warning("%s not in universe.yaml: market index guessed by correlation: %s",
                        company_id, guess)
            self._guessed[company_id] = guess
        return self._guessed[company_id]

    def index_ret(self, index_id: str | None) -> pd.Series:
        return self._index_ret.get(index_id, pd.Series(dtype=float))

    def j0(self, company_id: str, day: pd.Timestamp) -> pd.Timestamp:
        sessions = self.sessions(company_id)
        pos = sessions.searchsorted(pd.Timestamp(day).normalize())
        return sessions[pos] if pos < len(sessions) else pd.NaT


# ---- filters ---------------------------------------------------------------------------------

def _earnings_flags(events: pd.DataFrame, earnings: pd.DataFrame, markets: Markets,
                    k: int) -> pd.Series:
    """True if J0 is within `k` sessions of a release (release day aligned like J0)."""
    out = pd.Series(False, index=events.index)
    by_company = {c: g["date"].to_numpy() for c, g in earnings.groupby("company_id")}
    for cid, idx in events.groupby("company_id").groups.items():
        if cid not in by_company:
            continue
        sessions = markets.sessions(cid)
        release_pos = np.unique(sessions.searchsorted(by_company[cid]))
        j0 = events.loc[idx, "j0"]
        ok = j0.notna()
        pos = sessions.searchsorted(j0[ok].to_numpy())
        dist = np.abs(pos[:, None] - release_pos[None, :]).min(axis=1)
        out.loc[j0[ok].index] = dist <= k
    return out


def _company_z(events: pd.DataFrame, wiki_views: pd.DataFrame, params: Mapping) -> pd.Series:
    """Attention z-score of the company page on the shock day (NaN if no page)."""
    att = params["attention"]
    out = pd.Series(np.nan, index=events.index, name="z_company")
    for cid, idx in events.groupby("company_id").groups.items():
        page = wiki_views[wiki_views["page_id"] == company_page_id(cid)]
        if page.empty:
            continue
        series = page.set_index("date")["views_total"].sort_index()
        z = attention_z(series, att["window_days"], att["gap_days"])
        out.loc[idx] = z.reindex(events.loc[idx, "date_utc"]).to_numpy()
    return out


def _max_abs_market(events: pd.DataFrame, markets: Markets, lo: int, hi: int) -> pd.Series:
    """Largest absolute index return over sessions J0+lo .. J0+hi."""
    out = pd.Series(np.nan, index=events.index, name="max_abs_market")
    for i, ev in events.iterrows():
        if pd.isna(ev["j0"]):
            continue
        sessions = markets.sessions(ev["company_id"])
        p = sessions.searchsorted(ev["j0"])
        days = sessions[max(p + lo, 0): p + hi + 1]
        ret = markets.index_ret(markets.index_of(ev["company_id"])).reindex(days)
        out.loc[i] = ret.abs().max()
    return out


# ---- events ----------------------------------------------------------------------------------

def build_events(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    market: pd.DataFrame,
    earnings: pd.DataFrame,
    wiki_views: pd.DataFrame,
    gdelt_daily: pd.DataFrame,
    *,
    universe: Mapping[str, Company] | None = None,
    params: Mapping | None = None,
    factors_for=None,
) -> pd.DataFrame:
    """Detected shocks with sign, J0, CARs and exclusion flags (dossier §7.1-7.5).

    Returns the columns of the frozen `events` table except is_natural_experiment and split
    (added by ``finalize``), plus diagnostics: n_peak_days, n_est, beta, z_company,
    max_abs_market.

    Robustness knobs (§11.4) read from `params["returns"]`: `car_window` (window of the
    car_0_2 column, default [0, 2]) and the optional `model` ("market", "market_adjusted" or
    "ff3"; "ff3" needs `factors_for(company_id)`, see ``returns.compute_cars``).
    """
    params = params or load_params()
    flt = params["filters"]
    ev = detect_shocks(panel, params)
    if ev.empty:
        log.warning("no shock detected")
    ev["event_id"] = ev["person_id"] + "-" + pd.to_datetime(ev["date_utc"]).dt.strftime("%Y%m%d")

    neg = gdelt_daily.set_index(["person_id", "date"])["tone_neg_share"]
    keys = pd.MultiIndex.from_arrays([ev["person_id"], ev["date_utc"]])
    ev["tone_neg_share"] = neg.reindex(keys).to_numpy()
    ev["sign"] = sign_of(ev["tone_rel"], params["tone_sign_threshold"]).to_numpy()

    markets = Markets(prices, market, universe)
    ev["j0"] = pd.to_datetime(pd.Series(
        [markets.j0(c, d) for c, d in zip(ev["company_id"], ev["date_utc"], strict=True)],
        index=ev.index, dtype="datetime64[ns]"))

    has_j0 = ev["j0"].notna()
    mapping = {c: markets.index_of(c) for c in ev["company_id"].unique()}
    ret_params = params["returns"]
    model = ret_params.get("model", "market")
    # The target column keeps its name; robustness runs move its window (§11.4).
    windows = dict(CAR_WINDOWS, car_0_2=tuple(ret_params["car_window"]))
    cars = R.compute_cars(
        ev.loc[has_j0, ["event_id", "company_id", "j0"]], prices, market, mapping,
        model=model, windows=windows,
        estimation_window=tuple(ret_params["estimation_window"]),
        sessions_for=markets.sessions, factors_for=factors_for,
    )
    ev = ev.merge(cars[["event_id", *CAR_WINDOWS, "beta", "n_est"]], on="event_id", how="left")
    ev["n_est"] = ev["n_est"].fillna(0).astype(int)

    ev["z_company"] = _company_z(ev, wiki_views, params).to_numpy()
    car_lo, car_hi = params["returns"]["car_window"]
    ev["max_abs_market"] = _max_abs_market(ev, markets, car_lo, car_hi).to_numpy()

    # The market-adjusted model estimates nothing, so it has no history requirement.
    short = (ev["n_est"] < flt["min_estimation_obs"]) & (model != "market_adjusted")
    reasons = pd.DataFrame({
        "missing_returns": ev["car_0_2"].isna() & ~short,
        "short_history": short,
        "earnings": _earnings_flags(ev, earnings, markets,
                                    flt["earnings_window_sessions"]).to_numpy(),
        "company_dominant": (ev["z_company"] - ev["z_wiki"]) > flt["company_dominance_z_gap"],
        "extreme_market": ev["max_abs_market"] > flt["extreme_market_abs_ret"],
    })[list(EXCLUSION_REASONS)]
    ev["excluded"] = reasons.any(axis=1)
    names = np.array(reasons.columns)
    ev["exclusion_reason"] = pd.Series(
        [";".join(names[row]) or None for row in reasons.to_numpy(bool)],
        index=ev.index, dtype=object)
    if len(ev):
        log.info("shocks=%d excluded=%d (%s)", len(ev), int(ev["excluded"].sum()),
                 ", ".join(f"{k}={int(v)}" for k, v in reasons.sum().items()))

    return ev[["event_id", "person_id", "company_id", "date_utc", "j0", "z_wiki", "z_news",
               "n_articles", "tone_rel", "tone_neg_share", "sign", *CAR_WINDOWS, "excluded",
               "exclusion_reason", "n_peak_days", "n_est", "beta", "z_company",
               "max_abs_market"]]


# ---- events table (Q5) ------------------------------------------------------------------------

EVENT_COLUMNS = list(TABLES["events"].columns)


def flag_natural_experiments(events: pd.DataFrame, candidates: pd.DataFrame,
                             days: int = ne.EXCLUSION_DAYS) -> pd.Series:
    """True for events of a natural-experiment company within `days` days of it (§11.2).

    The natural experiment and its neighbours are kept out of training and calibration,
    whoever the event is about: a CFO shock two days after the CEO's arrest is contaminated.
    """
    out = pd.Series(False, index=events.index, name="is_natural_experiment")
    for cand in candidates.itertuples(index=False):
        same = events["company_id"] == cand.company_id
        near = (pd.to_datetime(events["date_utc"]) - cand.date).dt.days.abs() <= days
        out |= same & near
    return out


def finalize(events: pd.DataFrame, candidates: pd.DataFrame,
             params: Mapping | None = None) -> pd.DataFrame:
    """Add is_natural_experiment and split, keep the frozen columns, validate (§7.7)."""
    params = params or load_params()
    ev = events.copy()
    ev["is_natural_experiment"] = flag_natural_experiments(ev, candidates).to_numpy()
    ev = splits.assign_splits(ev, params["splits"], params["splits"]["gap_sessions"])
    ev["n_articles"] = ev["n_articles"].astype("int64")
    ev["excluded"] = ev["excluded"].astype(bool)
    ev = ev[EVENT_COLUMNS].sort_values(["date_utc", "person_id"]).reset_index(drop=True)
    return validate(ev, "events")


def _parse_overrides(pairs: list[str]) -> dict:
    """["shocks.z_wiki_min=2.5", ...] -> nested dict with YAML-parsed values."""
    out: dict = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if not key or not _:
            raise ValueError(f"--set expects key.path=value, got {pair!r}")
        node = out
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(value)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m kpr.build.events",
        description="Detect shocks, compute CARs and write the events table (dossier §7).",
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a quant.yaml parameter, e.g. shocks.z_wiki_min=2.5")
    parser.add_argument("--out", type=Path,
                        help="write a parquet file here instead of the shared events table")
    parser.add_argument("--dry-run", action="store_true", help="build and validate only")
    args = parser.parse_args(argv)

    params = load_params(_parse_overrides(args.set))
    tables = {n: read_table(n) for n in ("panel", "prices", "market", "earnings", "wiki_views",
                                         "gdelt_daily")}
    raw = build_events(**tables, params=params)
    events = finalize(raw, ne.load_candidates(), params)

    kept = events[~events["excluded"]]
    log.info("events=%d kept=%d reserved=%d", len(events), len(kept),
             int(events["is_natural_experiment"].sum()))
    print(kept.groupby(["split", "sign"]).size().unstack(fill_value=0).to_string())

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(args.out, index=False)
        log.info("wrote %s", args.out)
    elif args.dry_run or use_fixtures():
        log.info("not written (%s)", "dry run" if args.dry_run else
                 "fixtures: pass --out to save, the shared table is for real data only")
    elif args.set:
        log.info("not written: parameter overrides are for robustness runs, pass --out")
    else:
        log.info("wrote %s", write_table(events, "events"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
