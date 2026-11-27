"""Robustness runner over parameter variants.

Owner: quant session. Task Q9 in PARALLEL_SESSIONS.md §7. Dossier §11.4.

Each method choice is varied and the whole events pipeline is re-run. The ranking of
persons should stay broadly the same: the result is one table of rank correlations
(Spearman) with the reference version, and a variant that upsets the ranking is flagged.

Variants of the events pipeline (built from config/params/quant.yaml):

- shock threshold z_wiki: `robustness.z_wiki_min` (2.5 and 4 besides the reference 3);
- reaction window: `returns.robustness_windows` ([0, 1], [0, 5], [-1, 1] instead of [0, 2]);
- abnormal return model: market-adjusted, Fama-French 3 (only when factors are given);
- earnings filter: `robustness.earnings_window_sessions` (0, 2 besides 1) and no filter.

The variants of §11.4 that change the scores rather than the events (tone source, score
window, person selection) belong to the ml session's scoring (M10). The person score used
here is pluggable (``score_fn``); until M10 is merged, two event-level proxies are offered:
``sensitivity`` (per-person OLS slope of CAR on relative tone) and ``mean_abs_car``.
Scores use kept train + validation events only (§9.5).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from kpr.build import events as E
from kpr.utils.logging import get_logger
from kpr.validation import event_study as ES
from kpr.validation import natural_experiments as ne

log = get_logger(__name__)

TABLE_NAMES = ("panel", "prices", "market", "earnings", "wiki_views", "gdelt_daily")
# Below this rank correlation with the reference, a variant "upsets the ranking" (§11.4).
UPSET_SPEARMAN = 0.5
NO_FILTER = -1  # earnings_window_sessions value that disables the earnings filter


# ---- person scores ---------------------------------------------------------------------------

def sensitivity_score(events: pd.DataFrame, min_events: int = 5) -> pd.Series:
    """Per-person OLS slope of CAR on relative tone (kept train + validation events)."""
    ev = ES.select(events).dropna(subset=["car_0_2", "tone_rel"])
    out = {}
    for pid, g in ev.groupby("person_id"):
        if len(g) >= min_events and g["tone_rel"].std() > 0:
            out[pid] = float(np.polyfit(g["tone_rel"], g["car_0_2"], 1)[0])
    return pd.Series(out, name="score", dtype=float)


def mean_abs_car_score(events: pd.DataFrame, min_events: int = 3) -> pd.Series:
    """Per-person mean |CAR| (kept train + validation events)."""
    ev = ES.select(events).dropna(subset=["car_0_2"])
    g = ev.groupby("person_id")["car_0_2"]
    out = g.apply(lambda s: s.abs().mean())[g.size() >= min_events]
    return out.rename("score").astype(float)


SCORES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "sensitivity": sensitivity_score,
    "mean_abs_car": mean_abs_car_score,
}


# ---- variants ----------------------------------------------------------------------------------

def default_variants(params: Mapping | None = None) -> dict[str, dict]:
    """Variant name -> nested overrides of quant.yaml; "reference" has none."""
    params = params or E.load_params()
    rob, ret, flt = params["robustness"], params["returns"], params["filters"]
    variants: dict[str, dict] = {"reference": {}}
    for z in rob["z_wiki_min"]:
        if z != params["shocks"]["z_wiki_min"]:
            variants[f"z_wiki_{z:g}"] = {"shocks": {"z_wiki_min": z}}
    for lo, hi in ret["robustness_windows"]:
        name = f"window_{'m' if lo < 0 else ''}{abs(lo)}_{hi}"
        variants[name] = {"returns": {"car_window": [lo, hi]}}
    variants["market_adjusted"] = {"returns": {"model": "market_adjusted"}}
    variants["ff3"] = {"returns": {"model": "ff3"}}
    for k in rob["earnings_window_sessions"]:
        if k != flt["earnings_window_sessions"]:
            variants[f"earnings_{k}"] = {"filters": {"earnings_window_sessions": k}}
    variants["earnings_none"] = {"filters": {"earnings_window_sessions": NO_FILTER}}
    return variants


def run_variant(tables: Mapping[str, pd.DataFrame], overrides: Mapping, *, universe=None,
                candidates: pd.DataFrame | None = None, factors_for=None) -> pd.DataFrame:
    """The finalized events table under one set of parameter overrides."""
    params = E.load_params(overrides)
    raw = E.build_events(**{n: tables[n] for n in TABLE_NAMES}, universe=universe,
                         params=params, factors_for=factors_for)
    candidates = ne.load_candidates() if candidates is None else candidates
    return E.finalize(raw, candidates, params)


def run_all(tables: Mapping[str, pd.DataFrame], *, variants: Mapping[str, Mapping] | None = None,
            score_fn: Callable[[pd.DataFrame], pd.Series] = sensitivity_score, universe=None,
            factors_for=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every variant; return (summary table, person x variant score matrix)."""
    variants = dict(variants or default_variants())
    if "reference" not in variants:
        raise ValueError("variants must include 'reference'")
    candidates = ne.load_candidates()
    rows, scores = [], {}
    for name, overrides in variants.items():
        model = overrides.get("returns", {}).get("model")
        if model == "ff3" and factors_for is None:
            log.warning("variant %s skipped: no Fama-French factors (contract request open)",
                        name)
            rows.append({"variant": name, "note": "skipped: no FF3 factors"})
            continue
        events = run_variant(tables, overrides, universe=universe, candidates=candidates,
                             factors_for=factors_for)
        kept = ES.select(events)
        scores[name] = score_fn(events)
        by_sign = kept.groupby("sign")["car_0_2"].mean()
        rows.append({"variant": name, "events": len(events), "kept": len(kept),
                     "mean_car_neg": by_sign.get("negative", np.nan),
                     "mean_car_pos": by_sign.get("positive", np.nan),
                     "persons": int(scores[name].notna().sum()), "note": ""})
        log.info("variant %s: %d events, %d kept", name, len(events), len(kept))

    matrix = pd.DataFrame(scores)
    table = pd.DataFrame(rows).set_index("variant")
    ref = matrix["reference"]
    for name in matrix.columns:
        both = matrix[[name]].join(ref.rename("ref")).dropna()
        table.loc[name, "n_common"] = len(both)
        table.loc[name, "spearman"] = (both[name].corr(both["ref"], method="spearman")
                                       if len(both) > 2 else np.nan)
    table["upsets_ranking"] = table["spearman"] < UPSET_SPEARMAN
    counts = ["events", "kept", "persons", "n_common"]
    table[counts] = table[counts].astype("Int64")
    return table, matrix


def main(argv: list[str] | None = None) -> int:
    from kpr.utils.io import data_dir, read_table, use_fixtures

    parser = argparse.ArgumentParser(prog="python -m kpr.validation.robustness",
                                     description="Robustness table of dossier §11.4.")
    parser.add_argument("--score", choices=sorted(SCORES), default="sensitivity")
    parser.add_argument("--only", nargs="+", help="run only these variants (plus reference)")
    parser.add_argument("--out-dir", type=Path, help="default: processed/validation/")
    args = parser.parse_args(argv)

    tables = {n: read_table(n) for n in TABLE_NAMES}
    variants = default_variants()
    if args.only:
        unknown = sorted(set(args.only) - set(variants))
        if unknown:
            parser.error(f"unknown variants {unknown}; choose from {sorted(variants)}")
        variants = {k: v for k, v in variants.items() if k == "reference" or k in args.only}
    table, matrix = run_all(tables, variants=variants, score_fn=SCORES[args.score])
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(table.to_string())

    out_dir = args.out_dir
    if out_dir is None and not use_fixtures():
        out_dir = data_dir() / "processed" / "validation"
    if out_dir is None:
        log.info("fixtures: nothing written (pass --out-dir)")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "robustness.csv")
    matrix.to_csv(out_dir / "robustness_scores.csv")
    log.info("wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
