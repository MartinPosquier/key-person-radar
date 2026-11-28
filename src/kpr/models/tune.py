"""Hyperparameter tuning with time-series CV.

Owner: ml session. Task M7 in PARALLEL_SESSIONS.md §7. Dossier §9.4, §9.5.

The models of §9.4 (Lasso, Elastic Net, Random Forest, gradient boosting; logistic, forest
and boosting classifiers for the big-move task) are tuned by grid search on the training
split only, with the expanding-window yearly folds of kpr.models.cv. Every preprocessing
step lives inside the pipeline, so it is refitted on each training fold. Grids come from
`models.grids` in config/params/ml.yaml.

Run: python -m kpr.models.tune   (writes processed/models/tuning.json)
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from kpr.models.baselines import load_features
from kpr.models.cv import ExpandingYearSplit, load_ml_params, modelling_frame
from kpr.utils.io import data_dir, read_table, use_fixtures
from kpr.utils.logging import get_logger

log = get_logger(__name__)

SEED = 42  # dossier §13.4
REGRESSORS = ("lasso", "elasticnet", "rf", "hgb")
CLASSIFIERS = ("logit", "rf", "hgb")


class DropEmptyColumns(TransformerMixin, BaseEstimator):
    """Drop the columns with no observed value in the training fold (HGB cannot bin them)."""

    def fit(self, X, y=None):
        self.keep_ = ~np.all(np.isnan(np.asarray(X, dtype=float)), axis=0)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


def _prep(columns: list[str], scale: bool) -> ColumnTransformer:
    """Median imputation (+ standardisation for linear models), fitted on the fold only."""
    steps = [SimpleImputer(strategy="median", keep_empty_features=True)]
    if scale:
        steps.append(StandardScaler())
    return ColumnTransformer([("num", make_pipeline(*steps), columns)], remainder="drop")


def make_model(name: str, columns: list[str], task: str = "regression") -> Pipeline:
    """Untuned pipeline for a model of §9.4. HGB reads NaN natively, so it skips imputation."""
    regression = {
        "lasso": (Lasso(max_iter=20_000), True),
        "elasticnet": (ElasticNet(max_iter=20_000), True),
        "rf": (RandomForestRegressor(random_state=SEED, n_jobs=-1), False),
        "hgb": (HistGradientBoostingRegressor(random_state=SEED), None),
    }
    classification = {
        "logit": (LogisticRegression(solver="saga", max_iter=5_000), True),
        "rf": (RandomForestClassifier(random_state=SEED, n_jobs=-1,
                                      class_weight="balanced"), False),
        "hgb": (HistGradientBoostingClassifier(random_state=SEED), None),
    }
    table = regression if task == "regression" else classification
    if name not in table:
        raise KeyError(f"no {task} model {name!r}; known: {sorted(table)}")
    estimator, scale = table[name]
    prep = (ColumnTransformer([("num", DropEmptyColumns(), columns)], remainder="drop")
            if scale is None else _prep(columns, scale))
    return Pipeline([("prep", prep), ("model", estimator)])


def model_names(task: str) -> tuple[str, ...]:
    return REGRESSORS if task == "regression" else CLASSIFIERS


def grid_for(name: str, params: dict) -> dict[str, list]:
    """The grid of `models.grids.<name>`, with the pipeline prefix."""
    return {f"model__{k}": list(v) for k, v in params["models"]["grids"][name].items()}


def tune_model(model: BaseEstimator, grid: dict, X: pd.DataFrame, y: pd.Series,
               splitter: ExpandingYearSplit, scoring: str) -> dict:
    """Grid search on the CV folds. Without any fold, keep the first grid point (logged)."""
    if splitter.get_n_splits() == 0:
        best = {k: v[0] for k, v in grid.items()}
        log.warning("no CV fold available: keeping default %s", best)
        return {"best_params": best, "cv_score": np.nan, "n_folds": 0}
    search = GridSearchCV(model, grid, cv=splitter, scoring=scoring, n_jobs=1,
                          error_score="raise")
    search.fit(X, y)
    return {"best_params": search.best_params_, "cv_score": float(search.best_score_),
            "n_folds": splitter.get_n_splits()}


def tune_all(events: pd.DataFrame, features: pd.DataFrame, params: dict,
             targets: tuple[str, ...] = ("car_signed", "car_abs", "big_move")) -> dict:
    """Best hyperparameters per (target, model), tuned on the training split only."""
    feat_cols = [c for c in features.columns if c != "event_id"]
    train = modelling_frame(events, features, ("train",))
    cv = params["cv"]
    results: dict[str, dict] = {}
    for target in targets:
        task = "classification" if target == "big_move" else "regression"
        tr = train.dropna(subset=[target])
        if task == "classification" and tr[target].nunique() < 2:
            log.warning("%s: a single class in training, skipped", target)
            continue
        splitter = ExpandingYearSplit(tr["j0"], cv["first_train_end_year"], cv["gap_sessions"])
        y = tr[target].astype(int) if task == "classification" else tr[target]
        results[target] = {}
        for name in model_names(task):
            res = tune_model(make_model(name, feat_cols, task), grid_for(name, params),
                             tr[feat_cols], y, splitter, params["models"]["scoring"][task])
            results[target][name] = res
            log.info("%s / %s: %s", target, name, res)
    return results


def tuning_path():
    return data_dir() / "processed" / "models" / "tuning.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune the models of §9.4 by time-series CV")
    parser.add_argument("--first-train-end-year", type=int, default=None)
    args = parser.parse_args(argv)
    params = load_ml_params()
    if args.first_train_end_year:
        params["cv"]["first_train_end_year"] = args.first_train_end_year

    events = read_table("events")
    results = tune_all(events, load_features(events), params)
    if use_fixtures():
        log.info("fixture run: tuning results not written")
        return 0
    path = tuning_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    log.info("written to %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
