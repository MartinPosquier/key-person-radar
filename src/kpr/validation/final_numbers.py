"""Numbers quoted in docs/final/02_events_and_splits.md.

Owner: quant session. Task Q13 in PARALLEL_SESSIONS.md §7. Dossier §7, §9.5.

Every figure of file 02 comes from here, never typed by hand (docs/final/README.md). The
result is one flat JSON object whose keys are the placeholders of the file
(``{{n_shocks_raw}}`` -> ``"n_shocks_raw"``), plus the list ``natural_experiments`` for
section 2.5 and a ``_meta`` block.

- Filtering funnel (§7.4): the filters are applied in the order of the file. An event
  that fails several filters is counted at the first one it fails, so the "removed"
  column adds up to the detected total. ``n_flagged_*`` give the count of each filter on
  its own, to say which filter removed the most. Two steps come after the four of the
  file: missing returns (no price on a day of the window) and the 10-session gap at the
  split boundaries (``split_gap``).
- Event study (§7.6): mean CAR[0, 2] by sign on kept, non-reserved events. Test events
  are left out unless ``include_test`` is set: the file is filled in week 4, after the
  single test run (task M12), and nobody should read test-period CARs before that.
- Split sizes and comparability (§9.5): kept events per split, share of negative shocks,
  share of US companies and mean VIX on J0.

Usage: ``python scripts/quant_final_numbers.py [--include-test] [--out PATH]``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kpr.build import events as E
from kpr.build import splits as S
from kpr.utils.logging import get_logger
from kpr.validation import event_study as ES
from kpr.validation import natural_experiments as NE

log = get_logger(__name__)

REPO_ROOT = E.REPO_ROOT
DEFAULT_OUT = REPO_ROOT / "docs" / "final" / "numbers" / "quant.json"
# (placeholder suffix, exclusion reason), in the order of the funnel of file 02.
FUNNEL = (("earnings", "earnings"), ("company", "company_dominant"),
          ("market", "extreme_market"), ("history", "short_history"),
          ("returns", "missing_returns"), ("gap", S.GAP_REASON))
SIGN_KEYS = {"negative": "neg", "neutral": "neu", "positive": "pos"}
SPLIT_KEYS = {"train": "train", "validation": "val", "test": "test"}
N_CV_FOLDS = 4


def _num(x):
    """JSON-ready scalar: numpy types to Python, NaN to None."""
    if x is None:
        return None
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(x) else float(x)
    return x


def params_numbers(params: Mapping) -> dict:
    """param_*: the thresholds actually used (config/params/quant.yaml)."""
    sh, flt = params["shocks"], params["filters"]
    return {
        "param_z_wiki_min": sh["z_wiki_min"], "param_z_news_min": sh["z_news_min"],
        "param_min_articles": sh["min_articles"], "param_cluster_days": sh["cluster_days"],
        "param_attention_window_days": params["attention"]["window_days"],
        "param_attention_gap_days": params["attention"]["gap_days"],
        "param_tone_threshold": params["tone_sign_threshold"],
        "param_earnings_window": flt["earnings_window_sessions"],
        "param_company_dominance_z_gap": flt["company_dominance_z_gap"],
        "param_extreme_market": flt["extreme_market_abs_ret"],
        "param_min_estimation_obs": flt["min_estimation_obs"],
        "param_estimation_window": list(params["returns"]["estimation_window"]),
        "param_car_window": list(params["returns"]["car_window"]),
        "param_gap_sessions": params["splits"]["gap_sessions"],
    }


def _reasons(events: pd.DataFrame) -> pd.Series:
    """exclusion_reason as a set of reasons per event (the column joins them with ';')."""
    return events["exclusion_reason"].map(
        lambda r: set(r.split(";")) if isinstance(r, str) and r else set())


def funnel(events: pd.DataFrame) -> dict:
    """n_shocks_raw, n_after_*, n_removed_*, n_flagged_*, n_events_usable (§7.4)."""
    reasons = _reasons(events)
    unknown = set().union(*reasons) - {r for _, r in FUNNEL} if len(reasons) else set()
    if unknown:
        raise ValueError(f"exclusion reasons outside the funnel: {sorted(unknown)}")
    if (events["excluded"].astype(bool) != reasons.map(bool)).any():
        raise ValueError("excluded and exclusion_reason disagree")
    left = pd.Series(True, index=events.index)
    out = {"n_shocks_raw": len(events)}
    for key, reason in FUNNEL:
        hit = reasons.map(lambda rs, reason=reason: reason in rs)
        out[f"n_flagged_{key}"] = int(hit.sum())
        out[f"n_removed_{key}"] = int((left & hit).sum())
        left &= ~hit
        out[f"n_after_{key}"] = int(left.sum())
    out["n_events_usable"] = out[f"n_after_{FUNNEL[-1][0]}"]
    flagged = {k: out[f"n_flagged_{k}"] for k, _ in FUNNEL}
    out["largest_filter"] = max(flagged, key=flagged.get) if any(flagged.values()) else None
    return out


def sign_mix(events: pd.DataFrame) -> dict:
    """Share of each sign among all detected and among kept events (did filtering skew it?)."""
    out = {}
    for label, sub in (("raw", events), ("usable", events[~events["excluded"].astype(bool)])):
        share = sub["sign"].value_counts(normalize=True)
        for sign, key in SIGN_KEYS.items():
            out[f"share_{key}_{label}"] = _num(share.get(sign, 0.0)) if len(sub) else None
    return out


def group_sign_counts(events: pd.DataFrame, groups: Mapping[str, str]) -> dict:
    """n_media_*, n_control_*: kept, non-reserved events by group and sign."""
    kept = events[~events["excluded"].astype(bool) & (events["split"] != "reserved")]
    group = kept["company_id"].map(groups).fillna("unknown")
    out = {}
    for g in ("media", "control"):
        sub = kept[group == g]
        for sign, key in SIGN_KEYS.items():
            out[f"n_{g}_{key}"] = int((sub["sign"] == sign).sum())
        out[f"n_{g}"] = len(sub)
    out["n_unknown_group"] = int((group == "unknown").sum())
    return out


def event_study_numbers(events: pd.DataFrame, include_test: bool) -> dict:
    """car_mean_*, car_t_*, car_p_*, n_neg/n_neu/n_pos on kept events (§7.6)."""
    splits = ("train", "validation", "test") if include_test else ES.DEFAULT_SPLITS
    table = ES.mean_car_table(ES.select(events, splits))
    out = {"event_study_splits": list(splits)}
    for sign, key in SIGN_KEYS.items():
        row = table.loc[sign] if sign in table.index else None
        out[f"n_{key}"] = int(row["n"]) if row is not None else 0
        out[f"car_mean_{key}"] = _num(row["mean"]) if row is not None else None
        out[f"car_t_{key}"] = _num(row["t"]) if row is not None else None
        out[f"car_p_{key}"] = _num(row["p_t"]) if row is not None else None
    return out


def split_numbers(events: pd.DataFrame) -> dict:
    """n_train, n_validation, n_test, n_reserved (kept) and *_detected (all)."""
    kept = events[~events["excluded"].astype(bool)]
    out = {}
    for split in ("train", "validation", "test", "reserved"):
        out[f"n_{split}"] = int((kept["split"] == split).sum())
        out[f"n_{split}_detected"] = int((events["split"] == split).sum())
    return out


def cv_fold_numbers(events: pd.DataFrame, params: Mapping, n_folds: int = N_CV_FOLDS) -> dict:
    """Kept train events per expanding-window fold (validation years = last n_folds years).

    Counts before the inner gap that the ml session's CV harness drops at each fold
    boundary; the exact fold sizes used by the models are in numbers/ml.json.
    """
    kept = events[~events["excluded"].astype(bool) & (events["split"] == "train")]
    year = pd.to_datetime(kept["j0"].fillna(kept["date_utc"])).dt.year
    first, last = (pd.Timestamp(d).year for d in params["splits"]["train"])
    out = {}
    for k, val_year in enumerate(range(last - n_folds + 1, last + 1), start=1):
        out[f"cv_fold{k}_train_years"] = f"{first}-{val_year - 1}"
        out[f"cv_fold{k}_val_year"] = val_year
        out[f"n_cv_fold{k}_train"] = int((year < val_year).sum())
        out[f"n_cv_fold{k}_val"] = int((year == val_year).sum())
    return out


def load_regions(path: Path | None = None) -> dict[str, str]:
    """company_id -> 'US' | 'EU' from config/universe.yaml."""
    path = path or REPO_ROOT / "config" / "universe.yaml"
    with open(path, encoding="utf-8") as f:
        universe = yaml.safe_load(f) or {}
    return {c["company_id"]: c.get("region") for c in universe.get("companies") or []}


def comparability(events: pd.DataFrame, regions: Mapping[str, str],
                  vix: pd.DataFrame | None) -> dict:
    """share_neg_*, share_us_*, vix_* per split, on kept events (§9.5)."""
    kept = events[~events["excluded"].astype(bool)].copy()
    kept["region"] = kept["company_id"].map(regions)
    if vix is not None and len(vix):
        level = vix.set_index(pd.to_datetime(vix["date"]))["vix"].sort_index()
        day = pd.to_datetime(kept["j0"].fillna(kept["date_utc"]))
        # VIX of J0, or of the last day before it when J0 is a US holiday.
        kept["vix"] = level.asof(pd.DatetimeIndex(day)).to_numpy()
    else:
        kept["vix"] = np.nan
    out = {}
    for split, key in SPLIT_KEYS.items():
        sub = kept[kept["split"] == split]
        known = sub["region"].dropna()
        out[f"share_neg_{key}"] = _num((sub["sign"] == "negative").mean()) if len(sub) else None
        out[f"share_us_{key}"] = _num((known == "US").mean()) if len(known) else None
        out[f"vix_{key}"] = _num(sub["vix"].mean())
    return out


def natural_experiment_rows(candidates: pd.DataFrame, cars: pd.DataFrame | None,
                            universe_ids: Sequence[str]) -> tuple[list[dict], dict]:
    """Section 2.5: one row per candidate with its real CAR, and counts."""
    table = candidates.copy()
    if cars is not None and len(cars):
        table = table.merge(cars[["event_id", "car_0_2", "car_0_5"]], on="event_id", how="left")
    else:
        table["car_0_2"] = table["car_0_5"] = np.nan
    table["in_universe"] = table["company_id"].isin(set(universe_ids))
    rows = [{"event_id": r.event_id, "person_name": r.person_name, "company_id": r.company_id,
             "date": pd.Timestamp(r.date).date().isoformat(), "kind": r.kind,
             "wanted_by_company": _num(r.wanted_by_company), "in_universe": bool(r.in_universe),
             "car_0_2": _num(r.car_0_2), "car_0_5": _num(r.car_0_5)}
            for r in table.itertuples(index=False)]
    counts = {"n_natural_candidates": len(table),
              "n_natural_outside_universe": int((~table["in_universe"]).sum()),
              "n_natural_with_car": int(table["car_0_2"].notna().sum())}
    return rows, counts


def m12_run_date(decisions_dir: Path | None = None) -> str | None:
    """Date of the decisions/ml/ entry of task M12 (single test run), if any."""
    decisions_dir = decisions_dir or REPO_ROOT / "decisions" / "ml"
    dates = []
    for path in sorted(decisions_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        found = re.match(r"(\d{4}-\d{2}-\d{2})", path.name)
        if found and re.search(r"\bM12\b", text):
            dates.append(found.group(1))
    return dates[-1] if dates else None


def compute(events: pd.DataFrame, *, params: Mapping | None = None,
            groups: Mapping[str, str] | None = None, regions: Mapping[str, str] | None = None,
            vix: pd.DataFrame | None = None, candidates: pd.DataFrame | None = None,
            candidate_car: pd.DataFrame | None = None, include_test: bool = False,
            decisions_dir: Path | None = None) -> dict:
    """All numbers of file 02 as one JSON-ready dict."""
    params = params or E.load_params()
    groups = ES.company_groups() if groups is None else groups
    regions = load_regions() if regions is None else regions
    out: dict = {}
    out.update(params_numbers(params))
    out.update(funnel(events))
    out.update(sign_mix(events))
    out.update(group_sign_counts(events, groups))
    out.update(event_study_numbers(events, include_test))
    out.update(split_numbers(events))
    out.update(cv_fold_numbers(events, params))
    out.update(comparability(events, regions, vix))
    out["test_opened_on"] = m12_run_date(decisions_dir)
    if candidates is not None:
        rows, counts = natural_experiment_rows(candidates, candidate_car, list(regions))
        out.update(counts)
        out["natural_experiments"] = rows
    return {k: _num(v) for k, v in out.items()}


def main(argv: list[str] | None = None) -> int:
    from kpr.utils.io import read_table, use_fixtures

    parser = argparse.ArgumentParser(prog="python scripts/quant_final_numbers.py",
                                     description="Numbers of docs/final/02 (task Q13).")
    parser.add_argument("--include-test", action="store_true",
                        help="use test-period CARs in the event study (only after task M12)")
    parser.add_argument("--out", type=Path, help=f"default: {DEFAULT_OUT.relative_to(REPO_ROOT)}")
    args = parser.parse_args(argv)

    events = read_table("events")
    candidates = NE.load_candidates()
    try:
        prices, market = read_table("prices"), read_table("market")
        cars = NE.candidate_cars(candidates, prices, market, E.load_universe())
    except (FileNotFoundError, ValueError) as exc:
        log.warning("natural experiment CARs skipped: %s", exc)
        cars = None
    try:
        vix = read_table("vix")
    except FileNotFoundError:
        vix = None
    numbers = compute(events, vix=vix, candidates=candidates, candidate_car=cars,
                      include_test=args.include_test)
    numbers["_meta"] = {"generated_on": date.today().isoformat(), "fixtures": use_fixtures(),
                        "script": "scripts/quant_final_numbers.py"}
    text = json.dumps(numbers, indent=2, ensure_ascii=False)

    out = args.out
    if out is None and use_fixtures():
        print(text)
        log.info("fixtures: nothing written (pass --out); docs/final takes real numbers only")
        return 0
    out = out or DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    log.info("wrote %s (%d numbers)", out, len(numbers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
