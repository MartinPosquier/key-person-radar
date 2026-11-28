"""Owner: ml session. Tests for kpr.models.tune and kpr.models.train (task M7, §9.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kpr.models import train as TR
from kpr.models import tune as TU

PARAMS = {
    "cv": {"first_train_end_year": 2017, "gap_sessions": 10},
    "models": {
        "scoring": {"regression": "neg_mean_absolute_error",
                    "classification": "average_precision"},
        "selection_metric": "mae",
        "grids": {  # small grids: the real ones live in config/params/ml.yaml
            "lasso": {"alpha": [1e-4, 1e-2]},
            "elasticnet": {"alpha": [1e-4], "l1_ratio": [0.5]},
            "rf": {"n_estimators": [50], "max_depth": [3], "min_samples_leaf": [20]},
            "hgb": {"learning_rate": [0.1], "max_depth": [2], "max_iter": [50],
                    "l2_regularization": [0.0]},
            "logit": {"C": [0.1, 1.0], "l1_ratio": [0.0]},
        },
    },
}


def _synthetic(n: int = 1500, seed: int = 0):
    """Events 2015-2023; CAR reacts twice as much to bad tone as to good tone."""
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(rng.choice(pd.bdate_range("2015-01-01", "2023-12-31"), n))
    tone = rng.normal(0, 1, n)
    feats = pd.DataFrame({
        "event_id": [f"e{i}" for i in range(n)], "tone_rel": tone,
        "tone_neg": np.maximum(0, -tone), "tone_pos": np.maximum(0, tone),
        "z_wiki": rng.normal(5, 1, n), "vix": rng.normal(20, 5, n),
        "vol_20d": np.abs(rng.normal(0.015, 0.004, n)), "founder": rng.integers(0, 2, n) * 1.0,
    })
    car = -0.02 * feats["tone_neg"] + 0.01 * feats["tone_pos"] + rng.normal(0, 0.005, n)
    events = pd.DataFrame({
        "event_id": feats["event_id"], "company_id": rng.choice(["A", "B"], n), "j0": dates,
        "car_0_2": car, "excluded": False, "is_natural_experiment": False,
        "split": np.where(dates.year <= 2021, "train", "validation"),
    })
    return events, feats


@pytest.fixture(scope="module")
def data():
    return _synthetic()


@pytest.fixture(scope="module")
def tuning(data):
    return TU.tune_all(*data, PARAMS)


def test_every_model_of_the_dossier_builds():
    cols = ["a", "b"]
    for name in TU.REGRESSORS:
        assert TU.make_model(name, cols).named_steps["model"] is not None
    for name in TU.CLASSIFIERS:
        assert hasattr(TU.make_model(name, cols, "classification"), "predict_proba")
    with pytest.raises(KeyError):
        TU.make_model("lasso", cols, "classification")


def test_tuning_uses_the_cv_folds_and_the_grid(tuning):
    assert set(tuning) == {"car_signed", "car_abs", "big_move"}
    assert set(tuning["car_signed"]) == set(TU.REGRESSORS)
    assert set(tuning["big_move"]) == set(TU.CLASSIFIERS)
    lasso = tuning["car_signed"]["lasso"]
    assert lasso["n_folds"] == 4
    assert lasso["best_params"]["model__alpha"] in (1e-4, 1e-2)
    assert lasso["cv_score"] < 0  # negative MAE


def test_tuning_never_sees_validation_or_test(data, tuning):
    events, feats = data
    poisoned = events.copy()
    poisoned.loc[poisoned["split"] == "validation", "car_0_2"] = 1e3
    poisoned.loc[poisoned.index[:5], "split"] = "test"
    poisoned.loc[poisoned.index[:5], "car_0_2"] = -1e3
    clean = events.copy()
    clean.loc[clean.index[:5], "split"] = "test"
    a, b = TU.tune_all(poisoned, feats, PARAMS), TU.tune_all(clean, feats, PARAMS)
    for target in b:
        for name in b[target]:
            assert a[target][name]["best_params"] == b[target][name]["best_params"]
            # parallel tree fits may differ in the last float digit only
            assert a[target][name]["cv_score"] == pytest.approx(b[target][name]["cv_score"])


def test_models_beat_the_mean_on_validation(data, tuning):
    comparison, preds, fitted = TR.fit_and_compare(*data, PARAMS, tuning)
    signed = comparison[comparison["target"] == "car_signed"].set_index("model")
    for name in TU.REGRESSORS:
        assert signed.loc[name, "mae"] < 0.7 * signed.loc["B0", "mae"], name
    assert signed.loc[["B0", "B1", "B2"], "baseline"].all()
    assert set(preds["model"]) >= {"B0", "B1", "B2", *TU.REGRESSORS}
    assert ("car_signed", "hgb") in fitted and ("car_signed", "B1") not in fitted
    best = TR.choose_best(comparison)
    assert best["car_signed"] in TU.REGRESSORS and best["big_move"] in TU.CLASSIFIERS


def test_no_fold_keeps_the_first_grid_point():
    events, feats = _synthetic(n=200)
    events["j0"] = pd.Timestamp("2020-06-01")
    out = TU.tune_all(events, feats, PARAMS, targets=("car_signed",))
    assert out["car_signed"]["lasso"] == {"best_params": {"model__alpha": 1e-4},
                                          "cv_score": out["car_signed"]["lasso"]["cv_score"],
                                          "n_folds": 0}


def test_main_runs_on_fixtures(monkeypatch):
    monkeypatch.setenv("KPR_USE_FIXTURES", "1")
    assert TR.main([]) == 0
