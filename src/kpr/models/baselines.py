"""Baselines B0, B1, B2 and the time-series CV harness.

Owner: ml session. Task M3 in PARALLEL_SESSIONS.md §7. Dossier §9.3.

  B0  mean of the training target ("nothing special happens"). For the absolute target:
      the stock's own historical mean |CAR[0,2]| (training events of the same company,
      falling back to the overall mean for a company never seen in training).
  B1  linear model on the market-context features only (family E).
  B2  linear model on market context + shock intensity (families E and A), without tone
      or person characteristics.

H1 is judged against B1; the gain over B2 is the specific value of tone and person.
The classification task (big move, §9.1) gets the same three baselines with a prior-rate
dummy and logistic regressions. The CV harness itself lives in kpr.models.cv.

Run: python -m kpr.models.baselines   (KPR_USE_FIXTURES=1 to run on the fixtures)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from kpr.build.features import family_columns
from kpr.models.cv import (
    ExpandingYearSplit,
    fit_predict,
    load_ml_params,
    modelling_frame,
    oof_predictions,
)
from kpr.utils.io import data_dir, read_table, use_fixtures
from kpr.utils.logging import get_logger

log = get_logger(__name__)

BASELINE_FAMILIES = {"B1": ("E_market",), "B2": ("E_market", "A_intensity")}


class GroupMeanRegressor(BaseEstimator, RegressorMixin):
    """Predict the training mean of y within each value of `group_col` (global mean if unseen)."""

    def __init__(self, group_col: str = "company_id"):
        self.group_col = group_col

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.means_ = y.groupby(X[self.group_col]).mean().to_dict()
        self.global_mean_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.group_col].map(self.means_).fillna(self.global_mean_).to_numpy(float)


def _numeric(columns: list[str]) -> ColumnTransformer:
    """Impute (median) and standardise the given columns; both fitted on the fold only."""
    steps = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          StandardScaler())
    return ColumnTransformer([("num", steps, columns)], remainder="drop")


def baseline_columns(columns, name: str) -> list[str]:
    return [c for fam in BASELINE_FAMILIES[name] for c in family_columns(columns, fam)]


def make_baselines(columns, target: str = "car_signed") -> dict[str, BaseEstimator]:
    """B0, B1, B2 for a target of kpr.models.cv.TARGETS, given the feature columns."""
    cols = {name: baseline_columns(columns, name) for name in BASELINE_FAMILIES}
    if target == "big_move":
        return {
            "B0": DummyClassifier(strategy="prior"),
            **{name: Pipeline([("prep", _numeric(c)),
                               ("model", LogisticRegression(max_iter=1000))])
               for name, c in cols.items()},
        }
    b0 = GroupMeanRegressor() if target == "car_abs" else DummyRegressor(strategy="mean")
    return {
        "B0": b0,
        **{name: Pipeline([("prep", _numeric(c)), ("model", LinearRegression())])
           for name, c in cols.items()},
    }


def summarise(pred: pd.DataFrame, target: str) -> dict[str, float]:
    """Basic metrics of pooled predictions (the full metrics module is task M4)."""
    if pred.empty:
        return {"n": 0}
    y, p = pred["y_true"].to_numpy(float), pred["y_pred"].to_numpy(float)
    if target == "big_move":
        pr_auc = average_precision_score(y, p) if 0 < y.sum() < len(y) else np.nan
        return {"n": len(y), "pr_auc": pr_auc, "base_rate": float(y.mean())}
    return {"n": len(y), "mae": mean_absolute_error(y, p),
            "rmse": float(np.sqrt(np.mean((y - p) ** 2))), "r2": r2_score(y, p)}


def load_features(events: pd.DataFrame) -> pd.DataFrame:
    features = read_table("features")
    if not family_columns(features.columns, "E_market"):
        # The frozen fixture holds only 5 columns: rebuild the full matrix in memory.
        from kpr.build.features import build_features

        log.info("features table has no market columns: rebuilding them in memory")
        names = ("panel", "prices", "market", "vix", "mandates", "trends_ratio",
                 "wiki_views", "gdelt_daily")
        features = build_features(events, **{n: read_table(n) for n in names})
    return features


def run(events: pd.DataFrame, features: pd.DataFrame, params: dict) -> pd.DataFrame:
    """CV on the training split, then fit on train and score on validation, per target."""
    feat_cols = [c for c in features.columns if c != "event_id"]
    X_cols = [*feat_cols, "company_id"]
    train = modelling_frame(events, features, ("train",))
    valid = modelling_frame(events, features, ("validation",))
    cv = params["cv"]
    rows = []
    for target in ("car_signed", "car_abs", "big_move"):
        tr = train.dropna(subset=[target])
        va = valid.dropna(subset=[target])
        splitter = ExpandingYearSplit(tr["j0"], cv["first_train_end_year"], cv["gap_sessions"])
        proba = target == "big_move"
        for name, model in make_baselines(feat_cols, target).items():
            oof = oof_predictions(model, tr[X_cols], tr[target], splitter,
                                  ids=tr["event_id"], proba=proba)
            rows.append({"target": target, "model": name, "stage": "cv",
                         "folds": splitter.get_n_splits(), **summarise(oof, target)})
            if len(tr) and len(va):
                pred = fit_predict(model, tr[X_cols], tr[target], va[X_cols], proba)
                val = pd.DataFrame({"y_true": va[target].to_numpy(), "y_pred": pred})
                rows.append({"target": target, "model": name, "stage": "validation",
                             **summarise(val, target)})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Baselines B0-B2 (dossier §9.3)")
    parser.add_argument("--first-train-end-year", type=int, default=None,
                        help="override cv.first_train_end_year of config/params/ml.yaml")
    args = parser.parse_args(argv)
    params = load_ml_params()
    if args.first_train_end_year:
        params["cv"]["first_train_end_year"] = args.first_train_end_year

    events = read_table("events")
    results = run(events, load_features(events), params)
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        log.info("baselines:\n%s", results.round(5).to_string(index=False))
    if use_fixtures():
        log.info("fixture run: results not written")
        return 0
    out = data_dir() / "processed" / "models" / "baselines.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    log.info("written to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
