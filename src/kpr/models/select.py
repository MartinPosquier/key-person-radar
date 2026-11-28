"""Feature selection suite: filter, wrapper, embedded, stability.

Owner: ml session. Task M9 in PARALLEL_SESSIONS.md §7. Dossier §9.6.

Golden rule (LAB 4 Q5): selection happens inside the cross-validation, never before. Every
selector below is a scikit-learn transformer placed in a pipeline after imputation and
scaling, so `cv_selection` refits it on each training fold and scores it on the next year;
it also records how often each feature was kept across folds.

  1. filter     mutual information (SelectKBest), then mRMR to drop redundant features
  2. wrapper    RFE around a Random Forest
  3. embedded   Lasso regularisation path: the largest alpha at which a feature survives
  4. stability  Lasso on bootstrap resamples; a feature kept in > 80 % of fits is reliable

The descriptive tables (mutual information, Lasso path, stability) use the training split
only. Run: python -m kpr.models.select   (writes processed/models/selection*.csv)
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, lasso_path
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from kpr.build.features import FAMILIES, family_columns
from kpr.models.baselines import load_features
from kpr.models.cv import ExpandingYearSplit, load_ml_params, modelling_frame
from kpr.utils.io import data_dir, read_table, use_fixtures
from kpr.utils.logging import get_logger

log = get_logger(__name__)

SEED = 42


def _mi(X, y) -> np.ndarray:
    return mutual_info_regression(X, y, random_state=SEED)


class MRMRSelector(TransformerMixin, BaseEstimator):
    """Greedy minimum-redundancy maximum-relevance selection of `k` features.

    Relevance is the mutual information with the target; redundancy is the mean absolute
    Pearson correlation with the features already chosen (the MID criterion with a
    correlation proxy, cheap and stable on a few thousand rows).
    """

    def __init__(self, k: int = 10):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, float)
        relevance = _mi(X, y)
        with np.errstate(invalid="ignore", divide="ignore"):  # constant columns
            corr = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)))
        chosen = [int(np.argmax(relevance))]
        while len(chosen) < min(self.k, X.shape[1]):
            score = relevance - corr[:, chosen].mean(axis=1)
            score[chosen] = -np.inf
            chosen.append(int(np.argmax(score)))
        self.support_ = np.zeros(X.shape[1], bool)
        self.support_[chosen] = True
        self.order_ = chosen
        return self

    def get_support(self, indices: bool = False):
        return np.flatnonzero(self.support_) if indices else self.support_

    def transform(self, X):
        return np.asarray(X, float)[:, self.support_]


def make_selectors(n_features: int, params: dict) -> dict[str, Pipeline]:
    """One pipeline per selection method: impute -> scale -> select -> model."""
    sel = params["selection"]
    k = min(sel["k_features"], n_features)
    forest = RandomForestRegressor(n_estimators=sel["rf_trees"], min_samples_leaf=20,
                                   random_state=SEED, n_jobs=-1)
    lasso_alpha = params.get("lasso_alpha", 1e-3)
    selectors = {
        "filter_mi": (SelectKBest(_mi, k=k), LinearRegression()),
        "filter_mrmr": (MRMRSelector(k=k), LinearRegression()),
        "wrapper_rfe": (RFE(forest, n_features_to_select=k, step=0.2), LinearRegression()),
        "embedded_lasso": ("passthrough", Lasso(alpha=lasso_alpha, max_iter=20_000)),
        "all_features": ("passthrough", LinearRegression()),
    }
    return {
        name: Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()), ("select", selector), ("model", model)])
        for name, (selector, model) in selectors.items()
    }


def _kept(pipe: Pipeline, columns: list[str]) -> list[str]:
    step = pipe.named_steps["select"]
    if step == "passthrough":
        coef = pipe.named_steps["model"].coef_
        return [c for c, w in zip(columns, coef, strict=True) if w != 0]
    return [c for c, keep in zip(columns, step.get_support(), strict=True) if keep]


def cv_selection(X: pd.DataFrame, y: pd.Series, splitter, params: dict):
    """Fit every selector inside each training fold; validation MAE and keep frequency."""
    columns = list(X.columns)
    scores, counts = [], {name: pd.Series(0, index=columns) for name in make_selectors(
        len(columns), params)}
    folds = list(splitter.split(X))
    for name, pipe in make_selectors(len(columns), params).items():
        errors, n = [], []
        for tr, va in folds:
            fitted = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
            errors.append(mean_absolute_error(y.iloc[va], fitted.predict(X.iloc[va])))
            n.append(len(_kept(fitted, columns)))
            counts[name][_kept(fitted, columns)] += 1
        scores.append({"method": name, "folds": len(folds),
                       "cv_mae": float(np.mean(errors)) if errors else np.nan,
                       "mean_n_kept": float(np.mean(n)) if n else np.nan})
    freq = pd.DataFrame({f"{name}_fold_freq": c / max(len(folds), 1)
                         for name, c in counts.items()})
    return pd.DataFrame(scores), freq


def lasso_survival(X: pd.DataFrame, y: pd.Series, n_alphas: int = 50) -> pd.Series:
    """Largest alpha of the Lasso path at which each feature still has a non-zero weight."""
    Xs = StandardScaler().fit_transform(
        SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X))
    yc = y.to_numpy(float) - y.mean()
    alpha_max = max(np.abs(Xs.T @ yc).max() / len(yc), 1e-12)
    grid = np.logspace(np.log10(alpha_max), np.log10(alpha_max * 1e-3), n_alphas)
    alphas, coefs, _ = lasso_path(Xs, yc, alphas=grid)
    alive = np.abs(coefs) > 0  # (features, alphas), alphas decreasing
    survival = [alphas[row].max() if row.any() else 0.0 for row in alive]
    return pd.Series(survival, index=X.columns, name="lasso_survival_alpha")


def stability(X: pd.DataFrame, y: pd.Series, alpha: float, n_boot: int = 100,
              seed: int = SEED) -> pd.Series:
    """Share of bootstrap Lasso fits (scaled within each resample) that keep each feature."""
    rng = np.random.default_rng(seed)
    kept = np.zeros(X.shape[1])
    for _ in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        pipe = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("scale", StandardScaler()),
                         ("lasso", Lasso(alpha=alpha, max_iter=20_000))])
        pipe.fit(X.iloc[idx], y.iloc[idx])
        kept += pipe.named_steps["lasso"].coef_ != 0
    return pd.Series(kept / n_boot, index=X.columns, name="stability_freq")


def selection_report(X: pd.DataFrame, y: pd.Series, splitter, params: dict):
    """Per-feature table of the four methods, and the CV score of each selector."""
    sel = params["selection"]
    cv_scores, fold_freq = cv_selection(X, y, splitter, params)
    filled = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X)
    mrmr = MRMRSelector(k=X.shape[1]).fit(filled, y)
    table = pd.DataFrame({
        "family": [next((f for f in FAMILIES if c in family_columns([c], f)), "other")
                   for c in X.columns],
        "mutual_info": _mi(filled, y),
        "mrmr_rank": pd.Series(range(1, X.shape[1] + 1),
                               index=X.columns[mrmr.order_]).reindex(X.columns).to_numpy(),
    }, index=X.columns)
    table = table.join(lasso_survival(X, y)).join(
        stability(X, y, params.get("lasso_alpha", 1e-3), sel["stability_bootstrap"]))
    table["stable"] = table["stability_freq"] > sel["stability_threshold"]
    table = table.join(fold_freq).sort_values("stability_freq", ascending=False)
    log.info("stable features (> %.0f%% of %d bootstraps): %d of %d",
             100 * sel["stability_threshold"], sel["stability_bootstrap"],
             int(table["stable"].sum()), len(table))
    return table, cv_scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Feature selection suite (dossier §9.6)")
    parser.add_argument("--target", default="car_signed", choices=["car_signed", "car_abs"])
    args = parser.parse_args(argv)
    params = load_ml_params()
    tuning = data_dir() / "processed" / "models" / "tuning.json"
    if not use_fixtures() and tuning.exists():
        best = json.loads(tuning.read_text(encoding="utf-8"))[args.target]["lasso"]
        params["lasso_alpha"] = best["best_params"]["model__alpha"]
    events = read_table("events")
    features = load_features(events)
    train = modelling_frame(events, features, ("train",)).dropna(subset=[args.target])
    cols = [c for c in features.columns if c != "event_id"]
    splitter = ExpandingYearSplit(train["j0"], params["cv"]["first_train_end_year"],
                                  params["cv"]["gap_sessions"])
    table, cv_scores = selection_report(train[cols], train[args.target], splitter, params)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        log.info("selectors (CV):\n%s", cv_scores.round(5).to_string(index=False))
        log.info("features:\n%s", table.round(4).head(15).to_string())
    if use_fixtures():
        log.info("fixture run: selection tables not written")
        return 0
    out = data_dir() / "processed" / "models"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"selection_{args.target}.csv", index_label="feature")
    cv_scores.to_csv(out / f"selection_cv_{args.target}.csv", index=False)
    log.info("written to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
