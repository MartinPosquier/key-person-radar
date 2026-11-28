"""Sensitivities S- and S+: method A (simulation), method B (mixed model), agreement.

Owner: ml session. Task M10 in PARALLEL_SESSIONS.md §7. Dossier §3.1-3.2, §10.1-10.3.

Method A (§10.1): for each person p and year y, build p's average profile over the events of
the 3-year window [y-2, y], put a standard shock on it (z_wiki = 4, tone at p's own 10th or
90th percentile, other variables at their mean) and ask the best predictive model for the
CAR. |prediction| in % is the raw sensitivity. The interval comes from refitting the model
on bootstrap resamples of the training events (5th-95th percentiles).
Method B (§10.2): |(beta + b_p) x delta tone| from the mixed model of the same window, with
a normal 5th-95th interval from the slope's standard error.
Agreement (§10.3): Spearman correlation between the A and B rankings each year; above 0.6
the published raw score is the mean of A and B, otherwise B alone.

Run: python -m kpr.scores.sensitivity   (writes processed/scores/sensitivity.parquet)
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.base import clone

from kpr.build.features import load_companies
from kpr.models.baselines import load_features
from kpr.models.cv import load_ml_params, modelling_frame
from kpr.models.mixed import rolling_slopes
from kpr.models.tune import make_model
from kpr.utils.io import data_dir, read_table, use_fixtures
from kpr.utils.logging import get_logger

log = get_logger(__name__)

SEED = 42
Z90 = norm.ppf(0.95)  # 5th-95th percentile interval


# ---- profiles and standard shocks -----------------------------------------------------------


def window_profiles(frame: pd.DataFrame, feat_cols: list[str], years: list[int],
                    window: int) -> pd.DataFrame:
    """Mean features per (person, company, year) over the events of [y - window + 1, y].

    Also returns n_events and the person's own 10th / 90th tone percentiles on the window.
    """
    rows = []
    for year in years:
        sel = frame[frame["j0"].dt.year.between(year - window + 1, year)]
        for (pid, cid), g in sel.groupby(["person_id", "company_id"]):
            row = g[feat_cols].mean(numeric_only=True)
            row["person_id"], row["company_id"], row["year"] = pid, cid, year
            row["n_events"] = len(g)
            row["tone_q10"] = g["tone_rel"].quantile(0.10)
            row["tone_q90"] = g["tone_rel"].quantile(0.90)
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def shocked(profiles: pd.DataFrame, feat_cols: list[str], z_wiki: float,
            tone: pd.Series | float, specificity: float | None = None) -> pd.DataFrame:
    """Feature matrix of the profiles hit by a shock of intensity z_wiki and tone `tone`.

    Specificity stays at the profile's mean (§10.1) unless a value is given (departure
    shock of §10.4).
    """
    X = profiles[feat_cols].copy()
    tone = pd.Series(tone, index=X.index, dtype=float)
    if "z_wiki" in X:
        X["z_wiki"] = z_wiki
    if specificity is not None and "specificity" in X:
        X["specificity"] = specificity
    for col, value in (("tone_rel", tone), ("tone_neg", np.maximum(0.0, -tone)),
                       ("tone_pos", np.maximum(0.0, tone))):
        if col in X:
            X[col] = value
    if "tone_x_attention" in X and "attention_share" in X:
        X["tone_x_attention"] = tone * X["attention_share"]
    if "tone_x_founder" in X and "founder" in X:
        X["tone_x_founder"] = tone * X["founder"]
    return X


def standard_shocks(profiles: pd.DataFrame, feat_cols: list[str], params: dict):
    """(negative, positive) standard-shock matrices of §3.1."""
    z = params["standard_shock"]["z_wiki"]
    return (shocked(profiles, feat_cols, z, profiles["tone_q10"]),
            shocked(profiles, feat_cols, z, profiles["tone_q90"]))


# ---- method A -------------------------------------------------------------------------------


def bootstrap_predict(model, X_train: pd.DataFrame, y_train: pd.Series,
                      matrices: dict[str, pd.DataFrame], n_boot: int,
                      seed: int = SEED) -> dict[str, np.ndarray]:
    """Refit `model` on bootstrap resamples of the training events; predict each matrix.

    Returns {name: array of shape (n_boot, n_rows)}.
    """
    rng = np.random.default_rng(seed)
    out = {name: np.empty((n_boot, len(X))) for name, X in matrices.items()}
    for b in range(n_boot):
        idx = rng.integers(0, len(X_train), len(X_train))
        fitted = clone(model).fit(X_train.iloc[idx], y_train.iloc[idx])
        for name, X in matrices.items():
            out[name][b] = fitted.predict(X)
    return out


def method_a(model, profiles: pd.DataFrame, feat_cols: list[str], params: dict,
             X_train: pd.DataFrame | None = None, y_train: pd.Series | None = None,
             n_boot: int = 0) -> pd.DataFrame:
    """Raw S- and S+ in % of CAR, with bootstrap 5th-95th intervals when n_boot > 0."""
    neg, pos = standard_shocks(profiles, feat_cols, params)
    out = profiles[["person_id", "company_id", "year", "n_events"]].copy()
    out["a_minus"] = 100 * np.abs(model.predict(neg))
    out["a_plus"] = 100 * np.abs(model.predict(pos))
    if n_boot and X_train is not None:
        boot = bootstrap_predict(model, X_train, y_train, {"neg": neg, "pos": pos}, n_boot)
        for side, key in (("minus", "neg"), ("plus", "pos")):
            draws = 100 * np.abs(boot[key])
            out[f"a_{side}_lo"] = np.quantile(draws, 0.05, axis=0)
            out[f"a_{side}_hi"] = np.quantile(draws, 0.95, axis=0)
    return out


# ---- method B -------------------------------------------------------------------------------


def method_b(slopes: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """|slope x delta tone| in % of CAR, delta tone = the standard shock's tone parts."""
    m = profiles[["person_id", "company_id", "year", "tone_q10", "tone_q90"]].merge(
        slopes[["person_id", "year", "slope_neg", "slope_pos", "se_neg", "se_pos"]],
        on=["person_id", "year"], how="inner")
    d_neg, d_pos = np.maximum(0.0, -m["tone_q10"]), np.maximum(0.0, m["tone_q90"])
    out = m[["person_id", "company_id", "year"]].copy()
    for side, slope, se, delta in (("minus", "slope_neg", "se_neg", d_neg),
                                   ("plus", "slope_pos", "se_pos", d_pos)):
        centre, half = m[slope] * delta, Z90 * m[se] * delta
        out[f"b_{side}"] = 100 * np.abs(centre)
        lo, hi = 100 * np.abs(centre - half), 100 * np.abs(centre + half)
        crosses = np.sign(centre - half) != np.sign(centre + half)
        out[f"b_{side}_lo"] = np.where(crosses, 0.0, np.minimum(lo, hi))
        out[f"b_{side}_hi"] = np.maximum(lo, hi)
    return out


# ---- combination ----------------------------------------------------------------------------


def _rank_corr(x: pd.Series, y: pd.Series) -> float:
    """Spearman correlation, NaN when either ranking is constant."""
    if x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return float(spearmanr(x, y).statistic)


def combine(a: pd.DataFrame, b: pd.DataFrame, min_spearman: float) -> pd.DataFrame:
    """Per year: Spearman agreement of A and B; mean of both if it exceeds the threshold,
    otherwise method B alone (§10.3). Rows missing from B keep method A."""
    m = a.merge(b, on=["person_id", "company_id", "year"], how="left")
    parts = []
    for _year, g in m.groupby("year"):
        g = g.copy()
        both = g.dropna(subset=["b_minus", "b_plus"])
        rho = np.nan
        if len(both) >= 3:
            rhos = [_rank_corr(both[f"a_{s}"], both[f"b_{s}"]) for s in ("minus", "plus")]
            rhos = [r for r in rhos if not np.isnan(r)]
            rho = float(np.mean(rhos)) if rhos else np.nan
        g["method_agreement"] = rho
        use_mean = bool(rho > min_spearman) if not np.isnan(rho) else False
        for side in ("minus", "plus"):
            for suffix in ("", "_lo", "_hi"):
                a_col, b_col = f"a_{side}{suffix}", f"b_{side}{suffix}"
                a_val = g[a_col] if a_col in g else g[f"a_{side}"]
                b_val = g[b_col]
                chosen = (a_val + b_val) / 2 if use_mean else b_val
                g[f"s_{side}_raw{suffix}"] = chosen.fillna(a_val)
        g["method"] = np.where(g["b_minus"].isna(), "A", "A+B" if use_mean else "B")
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# ---- pipeline -------------------------------------------------------------------------------


def load_predictive_model(feat_cols: list[str], train: pd.DataFrame, target: str):
    """Best model of M7 (processed/models), or an untuned boosting model on fixtures."""
    models = None if use_fixtures() else data_dir() / "processed" / "models"
    best = models / "best_models.json" if models is not None else None
    if best is not None and best.exists():
        name = json.loads(best.read_text(encoding="utf-8"))[target]
        log.info("method A uses the M7 best model for %s: %s", target, name)
        return joblib.load(models / f"{target}__{name}.joblib")
    log.warning("no M7 model available: fitting an untuned boosting model (fixtures only)")
    return make_model("hgb", feat_cols).fit(train[feat_cols], train[target])


def shrink_method_a(a: pd.DataFrame, mandates: pd.DataFrame, companies: pd.DataFrame,
                    k: float) -> pd.DataFrame:
    """§10.5 step 2: shrink method A towards its peer group (the interval moves along)."""
    from kpr.scores.smoothing import peer_groups, shrink

    a = a.copy()
    groups = peer_groups(a, mandates, companies)
    for side in ("minus", "plus"):
        before = a[f"a_{side}"]
        a[f"a_{side}"] = shrink(before, a["n_events"], groups, a["year"], k)
        for suffix in ("_lo", "_hi"):
            if f"a_{side}{suffix}" in a:
                a[f"a_{side}{suffix}"] = (a[f"a_{side}{suffix}"] + a[f"a_{side}"]
                                          - before).clip(lower=0)
    return a


def build_sensitivities(events: pd.DataFrame, features: pd.DataFrame, params: dict,
                        splits: tuple[str, ...], model=None, n_boot: int | None = None,
                        mandates: pd.DataFrame | None = None,
                        companies: pd.DataFrame | None = None) -> pd.DataFrame:
    feat_cols = [c for c in features.columns if c != "event_id"]
    frame = modelling_frame(events, features, splits)
    train = modelling_frame(events, features, ("train",)).dropna(subset=["car_signed"])
    model = model or load_predictive_model(feat_cols, train, "car_signed")
    years = sorted(frame["j0"].dt.year.unique().tolist())
    window = params["scores"]["window_years"]
    profiles = window_profiles(frame, feat_cols, years, window)
    n_boot = params["scores"]["bootstrap_samples"] if n_boot is None else n_boot
    a = method_a(model, profiles, feat_cols, params, train[feat_cols], train["car_signed"],
                 n_boot)
    if mandates is not None and companies is not None:
        a = shrink_method_a(a, mandates, companies, params["scores"]["shrinkage_k"])
    slopes = rolling_slopes(frame, years, params)
    b = method_b(slopes, profiles) if len(slopes) else pd.DataFrame(
        columns=["person_id", "company_id", "year", *[f"b_{s}{x}" for s in ("minus", "plus")
                                                     for x in ("", "_lo", "_hi")]])
    out = combine(a, b, params["scores"]["agreement_min_spearman"])
    log.info("sensitivities: %d person-years; method used: %s", len(out),
             out["method"].value_counts().to_dict())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S- and S+ by methods A and B (§10.1-10.3)")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"],
                        help="event splits used (add test only after the M12 go)")
    parser.add_argument("--n-boot", type=int, default=None)
    args = parser.parse_args(argv)
    params = load_ml_params()
    events = read_table("events")
    out = build_sensitivities(events, load_features(events), params, tuple(args.splits),
                              n_boot=args.n_boot, mandates=read_table("mandates"),
                              companies=load_companies())
    if use_fixtures():
        log.info("fixture run: sensitivities not written")
        return 0
    path = data_dir() / "processed" / "scores" / "sensitivity.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    log.info("written to %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
