"""Mixed-effects model with person-level tone slopes.

Owner: ml session. Task M8 in PARALLEL_SESSIONS.md §7. Dossier §9.4, §10.2.

    CAR_e = gamma' x_e + (beta- + b-_p) tone-_e + (beta+ + b+_p) tone+_e + eps_e

beta- and beta+ are the average sensitivities to the negative and positive parts of the tone;
b-_p and b+_p are person p's random deviations, shrunk towards zero when p has few events
(the Bayesian shrinkage of §3.4, for free). x_e are fixed-effect controls (shock intensity,
market context), median-imputed and standardised on the estimation sample. The person
slopes beta + b_p feed method B of the scores (task M10); `rolling_slopes` re-estimates the
model on each 3-year window to get one slope per person and year (§10.2).

Run: python -m kpr.models.mixed   (writes processed/models/mixed_slopes.parquet)
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from kpr.models.baselines import load_features
from kpr.models.cv import load_ml_params, modelling_frame
from kpr.utils.io import data_dir, read_table, use_fixtures
from kpr.utils.logging import get_logger

log = get_logger(__name__)

TONE = ("tone_neg", "tone_pos")


def prepare(frame: pd.DataFrame, controls: list[str], min_events: int) -> pd.DataFrame:
    """Estimation sample: persons with enough events, controls imputed and standardised."""
    counts = frame["person_id"].value_counts()
    keep = counts[counts >= min_events].index
    data = frame[frame["person_id"].isin(keep)].copy()
    log.info("mixed model sample: %d events, %d persons (dropped %d events of %d persons "
             "with fewer than %d events)", len(data), len(keep), len(frame) - len(data),
             len(counts) - len(keep), min_events)
    used = []
    for col in controls:
        if col not in data.columns or data[col].notna().sum() == 0:
            continue
        x = data[col].fillna(data[col].median())
        if x.std() > 0:
            data[col] = (x - x.mean()) / x.std()
            used.append(col)
    data.attrs["controls"] = used
    return data


def fit_mixed(data: pd.DataFrame, target: str = "car_signed", random_intercept: bool = False):
    """Fit the §9.4 model by REML; returns the statsmodels result (`converged` attribute)."""
    controls = data.attrs.get("controls", [])
    formula = f"{target} ~ " + " + ".join([*TONE, *controls])
    re_formula = ("1 + " if random_intercept else "0 + ") + " + ".join(TONE)
    model = smf.mixedlm(formula, data, groups=data["person_id"], re_formula=re_formula)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        result = model.fit(reml=True, method=["lbfgs", "powell"])
    for w in caught:
        if issubclass(w.category, ConvergenceWarning):
            log.warning("mixed model: %s", w.message)
    return result


def person_slopes(result, data: pd.DataFrame) -> pd.DataFrame:
    """Per person: total slopes beta + b_p on tone- and tone+, the random parts, n_events.

    se_neg / se_pos combine the variance of the fixed slope and the conditional variance of
    the person's random deviation (independence assumed): the basis of method B's interval.
    """
    fixed = result.fe_params
    fixed_var = np.diag(result.cov_params().loc[list(TONE), list(TONE)])
    rows = []
    for person, effects in result.random_effects.items():
        b_neg = float(effects.get("tone_neg", 0.0))
        b_pos = float(effects.get("tone_pos", 0.0))
        cond = result.random_effects_cov.get(person)
        cond_var = [float(cond.loc[t, t]) if cond is not None and t in cond.index else 0.0
                    for t in TONE]
        rows.append({
            "person_id": person,
            "slope_neg": fixed["tone_neg"] + b_neg, "slope_pos": fixed["tone_pos"] + b_pos,
            "se_neg": float(np.sqrt(fixed_var[0] + cond_var[0])),
            "se_pos": float(np.sqrt(fixed_var[1] + cond_var[1])),
            "b_neg": b_neg, "b_pos": b_pos,
            "n_events": int((data["person_id"] == person).sum()),
        })
    out = pd.DataFrame(rows)
    out.attrs["beta_neg"], out.attrs["beta_pos"] = fixed["tone_neg"], fixed["tone_pos"]
    out.attrs["converged"] = bool(result.converged)
    return out


def rolling_slopes(frame: pd.DataFrame, years: list[int], params: dict,
                   target: str = "car_signed") -> pd.DataFrame:
    """One fit per 3-year window [y - window + 1, y]: slopes per (person, year) (§10.2)."""
    cfg, window = params["mixed"], params["scores"]["window_years"]
    parts = []
    for year in years:
        sel = frame[frame["j0"].dt.year.between(year - window + 1, year)]
        data = prepare(sel, cfg["controls"], cfg["min_events_per_person"])
        if data["person_id"].nunique() < 2:
            log.warning("year %d: fewer than 2 persons in the window, skipped", year)
            continue
        result = fit_mixed(data, target, cfg["random_intercept"])
        slopes = person_slopes(result, data)
        slopes["year"] = year
        slopes["beta_neg"], slopes["beta_pos"] = slopes.attrs["beta_neg"], slopes.attrs[
            "beta_pos"]
        slopes["converged"] = slopes.attrs["converged"]
        parts.append(slopes)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mixed-effects tone slopes (dossier §10.2)")
    parser.add_argument("--splits", nargs="+", default=["train"],
                        help="event splits to estimate on (test only after the M12 go)")
    args = parser.parse_args(argv)
    params = load_ml_params()
    events = read_table("events")
    frame = modelling_frame(events, load_features(events), tuple(args.splits))
    years = sorted(frame["j0"].dt.year.unique().tolist())
    slopes = rolling_slopes(frame, years, params)
    if slopes.empty:
        log.warning("no window could be estimated")
        return 0
    summary = slopes.groupby("year")[["beta_neg", "beta_pos", "converged"]].first()
    log.info("average slopes per window:\n%s", summary.to_string())
    log.info("asymmetry |beta-| / |beta+| (latest window): %.2f",
             abs(summary["beta_neg"].iloc[-1]) / max(abs(summary["beta_pos"].iloc[-1]), 1e-12))
    if use_fixtures():
        log.info("fixture run: slopes not written")
        return 0
    out = data_dir() / "processed" / "models" / "mixed_slopes.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    slopes.to_parquet(out, index=False)
    log.info("written to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
