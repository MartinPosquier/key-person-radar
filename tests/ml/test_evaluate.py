"""Owner: ml session. Tests for kpr.models.evaluate (task M4, dossier §9.7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kpr.models import evaluate as E


def test_point_metrics_by_hand():
    y = np.array([1.0, -1.0, 2.0, 0.0])
    p = np.array([0.5, -1.0, 1.0, 0.5])
    assert E.mae(y, p) == pytest.approx(0.5)
    assert E.rmse(y, p) == pytest.approx(np.sqrt((0.25 + 0 + 1 + 0.25) / 4))
    assert E.r2_oos(y, p, benchmark=0.0) == pytest.approx(1 - 1.5 / 6)
    assert E.r2_oos(y, y) == 1.0
    assert E.sign_hit_rate(y, p) == pytest.approx(0.75)
    assert E.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert np.isnan(E.spearman([1, 2, 3], [5, 5, 5]))


def test_r2_oos_is_negative_when_worse_than_benchmark():
    y = np.array([0.01, -0.01, 0.02])
    assert E.r2_oos(y, -y) < 0


def test_classification_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.8, 0.9])
    assert E.pr_auc(y, p) == pytest.approx(1.0)
    assert E.recall(y, p) == 1.0
    assert E.f1(y, p) == pytest.approx(0.8)
    assert np.isnan(E.pr_auc([0, 0], [0.1, 0.2]))


def test_bootstrap_interval_contains_point_and_is_reproducible():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 300)
    p = y + rng.normal(0, 1, 300)
    point, lo, hi = E.bootstrap_ci(y, p, E.mae, n_boot=500)
    assert lo < point < hi
    assert (point, lo, hi) == E.bootstrap_ci(y, p, E.mae, n_boot=500)
    # the interval narrows with more events
    y2, p2 = np.tile(y, 4), np.tile(p, 4)
    _, lo2, hi2 = E.bootstrap_ci(y2, p2, E.mae, n_boot=500)
    assert hi2 - lo2 < hi - lo


def test_evaluate_regression_table_and_neutral_filter():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 0.02, 200)
    p = 0.5 * y + rng.normal(0, 0.01, 200)
    sign = np.where(np.abs(y) < 0.005, "neutral", np.where(y < 0, "negative", "positive"))
    table = E.evaluate(y, p, sign=sign, n_boot=200).set_index("metric")
    assert set(table.index) == {"mae", "rmse", "r2_oos", "spearman", "sign_hit_rate"}
    assert table.loc["sign_hit_rate", "n"] == int((sign != "neutral").sum())
    assert table.loc["spearman", "lo"] > 0.3
    assert (table["lo"] <= table["value"]).all() and (table["value"] <= table["hi"]).all()


def test_evaluate_classification_table():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200)
    table = E.evaluate(y, y * 0.6 + rng.uniform(0, 0.4, 200), task="classification",
                       n_boot=100)
    assert list(table["metric"]) == ["pr_auc", "f1", "recall"]
    with pytest.raises(ValueError):
        E.evaluate(y, y, task="ranking")


def test_paired_bootstrap_gain_detects_a_better_model():
    rng = np.random.default_rng(4)
    y = rng.normal(0, 1, 500)
    good, bad = y + rng.normal(0, 0.5, 500), y + rng.normal(0, 1.5, 500)
    better = E.paired_bootstrap_gain(y, good, bad, n_boot=500)
    assert better["gain"] > 0 and better["lo"] > 0 and better["p_value"] < 0.01
    same = E.paired_bootstrap_gain(y, good, good, n_boot=200)
    assert same["gain"] == 0 and same["p_value"] == 1.0


def test_main_on_a_predictions_file(tmp_path):
    rng = np.random.default_rng(5)
    y = rng.normal(0, 0.02, 50)
    path = tmp_path / "pred.csv"
    pd.DataFrame({"y_true": y, "y_pred": y * 0.5, "y_base": 0.0}).to_csv(path, index=False)
    assert E.main([str(path), "--n-boot", "50"]) == 0
