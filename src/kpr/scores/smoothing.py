"""0-10 scale, shrinkage, yearly smoothing and succession resets; writes the scores table.

Owner: ml session. Task M10 in PARALLEL_SESSIONS.md §7. Dossier §3.3-3.4, §10.5-10.6.

- Scale (§3.3): fixed thresholds = the 0th, 10th, ..., 100th percentiles of the raw
  sensitivities of 2015-2021, computed once and saved in config/score_scale.yaml, never
  recomputed (unless --recompute-scale). Raw values map linearly between thresholds; values
  above the last one are capped at 10 (the app flags them).
- Shrinkage (§3.4, §10.5): S~ = n/(n+k) S + k/(n+k) mean of the peer group (role x sector)
  that year, k = 10. Applied to method A only (the mixed model already shrinks method B).
- Smoothing (§3.4): S_pub = clip(w S~ + (1-w) S_pub(y-1), S_pub(y-1) +/- 2), w = 0.4.
- Succession (§3.4): in the year a person's mandate starts, the previous value is the peer
  group mean and the yearly cap does not apply; succession_flag marks that year.

Run: python -m kpr.scores.smoothing   (writes the `scores` table)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kpr.utils.logging import get_logger

log = get_logger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
SCALE_PATH = CONFIG_DIR / "score_scale.yaml"
SCALE_YEARS = (2015, 2021)  # training period of §9.5
TOP = 10.0


# ---- peer groups and shrinkage ----------------------------------------------------------------


def peer_groups(rows: pd.DataFrame, mandates: pd.DataFrame,
                companies: pd.DataFrame) -> pd.Series:
    """'role|sector' per row, the role being the mandate active at the end of the year."""
    sector = companies.drop_duplicates("company_id").set_index("company_id")["sector"]
    out = []
    for r in rows.itertuples(index=False):
        end = pd.Timestamp(year=int(r.year), month=12, day=31)
        m = mandates[(mandates["person_id"] == r.person_id)
                     & (mandates["company_id"] == r.company_id) & (mandates["start"] <= end)]
        role = m.sort_values("start")["role"].iloc[-1] if len(m) else "other"
        out.append(f"{role}|{sector.get(r.company_id, 'unknown')}")
    return pd.Series(out, index=rows.index, name="group")


def shrink(values: pd.Series, n: pd.Series, groups: pd.Series, years: pd.Series,
           k: float) -> pd.Series:
    """Bayesian shrinkage towards the (group, year) mean, weight n / (n + k)."""
    group_mean = values.groupby([groups, years]).transform("mean")
    w = n / (n + k)
    return w * values + (1 - w) * group_mean


# ---- fixed 0-10 scale ---------------------------------------------------------------------------


def compute_thresholds(raw: pd.Series) -> list[float]:
    return [float(q) for q in np.quantile(raw.dropna(), np.linspace(0, 1, 11))]


def to_scale(raw, thresholds: list[float]) -> np.ndarray:
    """Piecewise-linear map of raw sensitivities onto 0-10 (clamped at both ends)."""
    t = np.maximum.accumulate(np.asarray(thresholds, float))  # strictly usable for interp
    return np.interp(np.asarray(raw, float), t, np.linspace(0, TOP, len(t)))


def load_or_compute_scale(sens: pd.DataFrame, recompute: bool = False, write: bool = True,
                          path: Path = SCALE_PATH) -> dict:
    """Thresholds of config/score_scale.yaml; computed once on 2015-2021 if not yet done."""
    current = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    if current.get("computed") and not recompute:
        return current
    period = sens[sens["year"].between(*SCALE_YEARS)]
    if period.empty:
        period = sens
        log.warning("no person-year in %s-%s: scale computed on all years", *SCALE_YEARS)
    scale = {"computed": True, "period": list(SCALE_YEARS),
             "s_minus_thresholds": compute_thresholds(period["s_minus_raw"]),
             "s_plus_thresholds": compute_thresholds(period["s_plus_raw"])}
    if write:
        header = ("# Owner: ml session (task M10). Fixed 0-10 thresholds computed once on "
                  "2015-2021, then frozen.\n")
        path.write_text(header + yaml.safe_dump(scale, sort_keys=False), encoding="utf-8")
        log.info("score scale computed and frozen in %s", path)
    return scale


# ---- smoothing ----------------------------------------------------------------------------------


def smooth(rows: pd.DataFrame, column: str, omega: float, cap: float) -> pd.Series:
    """Yearly smoothing and cap per (person, company); reset at successions.

    `rows` needs: person_id, company_id, year, `column` (0-10), group_mean, succession_flag.
    """
    out = pd.Series(np.nan, index=rows.index)
    for _key, g in rows.sort_values("year").groupby(["person_id", "company_id"]):
        prev = None
        for idx, r in g.iterrows():
            target = r[column]
            if r["succession_flag"]:
                value = omega * target + (1 - omega) * r["group_mean"]  # no cap
            elif prev is None:
                value = target
            else:
                value = float(np.clip(omega * target + (1 - omega) * prev,
                                      prev - cap, prev + cap))
            out[idx] = value
            prev = value
    return out


def succession_flags(rows: pd.DataFrame, mandates: pd.DataFrame) -> pd.Series:
    """True in the year a mandate starts, if that year is not the first year scored overall."""
    first_year = rows["year"].min()
    starts = mandates.assign(year=mandates["start"].dt.year)[["person_id", "company_id",
                                                              "year"]]
    starts = starts[starts["year"] > first_year].drop_duplicates()
    flagged = rows[["person_id", "company_id", "year"]].merge(starts.assign(flag=True),
                                                              how="left")["flag"]
    return flagged.eq(True).set_axis(rows.index)


# ---- assembly -----------------------------------------------------------------------------------


def build_scores(sens: pd.DataFrame, value: pd.DataFrame, mandates: pd.DataFrame,
                 companies: pd.DataFrame, params: dict, scale: dict) -> pd.DataFrame:
    """The `scores` table of §10.6 from the raw sensitivities and values."""
    sc = params["scores"]
    rows = sens.merge(value, on=["person_id", "company_id", "year"], how="left")
    rows["succession_flag"] = succession_flags(rows, mandates)
    groups = peer_groups(rows, mandates, companies)
    out = rows[["person_id", "company_id", "year"]].copy()
    for side in ("minus", "plus"):
        thresholds = scale[f"s_{side}_thresholds"]
        scaled = pd.Series(to_scale(rows[f"s_{side}_raw"], thresholds), index=rows.index)
        work = rows[["person_id", "company_id", "year", "succession_flag"]].assign(
            target=scaled, group_mean=scaled.groupby([groups, rows["year"]]).transform("mean"))
        published = smooth(work, "target", sc["smoothing_omega"], sc["max_yearly_change"])
        lo = to_scale(rows[f"s_{side}_raw_lo"], thresholds)
        hi = to_scale(rows[f"s_{side}_raw_hi"], thresholds)
        out[f"s_{side}_raw"] = rows[f"s_{side}_raw"]
        out[f"s_{side}"] = published
        out[f"s_{side}_lo"] = np.minimum(lo, published)  # the interval always holds the score
        out[f"s_{side}_hi"] = np.maximum(hi, published)
    out["asymmetry"] = (rows["s_minus_raw"] / rows["s_plus_raw"]).where(rows["s_plus_raw"] > 0)
    for col in ("v_pct", "v_usd", "v_lo", "v_hi", "pi_share", "method_agreement"):
        out[col] = pd.to_numeric(rows[col], errors="coerce") if col in rows else np.nan
        out[col] = out[col].astype(float)
    out["n_events"] = rows["n_events"].astype("int64")
    out["succession_flag"] = rows["succession_flag"].astype(bool)
    out["year"] = out["year"].astype("int64")
    cols = ["person_id", "company_id", "year", "s_minus_raw", "s_plus_raw", "s_minus",
            "s_plus", "s_minus_lo", "s_minus_hi", "s_plus_lo", "s_plus_hi", "asymmetry",
            "v_pct", "v_usd", "v_lo", "v_hi", "pi_share", "n_events", "method_agreement",
            "succession_flag"]
    return out[cols].sort_values(["person_id", "year"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    from kpr.build.features import load_companies
    from kpr.models.baselines import load_features
    from kpr.models.cv import load_ml_params
    from kpr.scores.sensitivity import build_sensitivities
    from kpr.scores.value import build_values, load_departures
    from kpr.utils.io import data_dir, read_table, use_fixtures, write_table

    parser = argparse.ArgumentParser(description="0-10 scale and smoothing; scores table")
    parser.add_argument("--recompute-scale", action="store_true",
                        help="recompute the frozen thresholds (only with a written reason)")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    args = parser.parse_args(argv)
    params = load_ml_params()
    mandates, companies = read_table("mandates"), load_companies()
    if use_fixtures():  # compute the intermediate tables in memory, small bootstraps
        events = read_table("events")
        features = load_features(events)
        sens = build_sensitivities(events, features, params, tuple(args.splits), n_boot=5,
                                   mandates=mandates, companies=companies)
        tables = {n: read_table(n) for n in ("prices", "market")}
        value = build_values(events, features, params, tuple(args.splits), tables,
                             load_departures(), companies, n_model_boot=5)
    else:
        scores_dir = data_dir() / "processed" / "scores"
        sens = pd.read_parquet(scores_dir / "sensitivity.parquet")
        value = pd.read_parquet(scores_dir / "value.parquet")
    scale = load_or_compute_scale(sens, args.recompute_scale, write=not use_fixtures())
    scores = build_scores(sens, value, mandates, companies, params, scale)
    log.info("scores: %d person-years, %d with fewer than 8 events (low confidence)",
             len(scores), int((scores["n_events"] < 8).sum()))
    if use_fixtures():
        from kpr.schemas import validate

        validate(scores, "scores")
        log.info("fixture run: scores valid, not written")
        return 0
    log.info("written to %s", write_table(scores, "scores"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
