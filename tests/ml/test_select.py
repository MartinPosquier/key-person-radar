"""Owner: ml session. Tests for kpr.models.select (task M9, dossier §9.6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kpr.models import select as S
from kpr.models.cv import ExpandingYearSplit

PARAMS = {
    "selection": {"stability_bootstrap": 30, "stability_threshold": 0.8, "k_features": 3,
                  "rf_trees": 30},
    "lasso_alpha": 1e-3,
}


def _synthetic(n: int = 900, seed: int = 0):
    """3 informative features, a near-duplicate of the strongest one, 5 pure-noise columns."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "tone_neg": np.abs(rng.normal(0, 1, n)), "z_wiki": rng.normal(0, 1, n),
        "vix": rng.normal(0, 1, n),
        **{f"noise{i}": rng.normal(0, 1, n) for i in range(5)},
    })
    X["tone_neg_copy"] = X["tone_neg"] + rng.normal(0, 0.05, n)
    y = (-0.02 * X["tone_neg"] + 0.01 * X["z_wiki"] + 0.005 * X["vix"]
         + rng.normal(0, 0.005, n))
    dates = pd.Series(pd.to_datetime(rng.choice(pd.bdate_range("2015-01-01", "2021-12-31"), n)))
    order = dates.sort_values().index
    return X.loc[order].reset_index(drop=True), y.loc[order].reset_index(drop=True), \
        dates.loc[order].reset_index(drop=True)


@pytest.fixture(scope="module")
def report():
    X, y, dates = _synthetic()
    splitter = ExpandingYearSplit(dates, 2017, 10)
    return S.selection_report(X, y, splitter, PARAMS)


def test_mrmr_skips_the_redundant_copy():
    X, y, _ = _synthetic()
    sel = S.MRMRSelector(k=3).fit(X.to_numpy(), y)
    names = list(X.columns[sel.order_])
    assert names[0] in ("tone_neg", "tone_neg_copy")
    assert not {"tone_neg", "tone_neg_copy"} <= set(names)  # never both
    assert "z_wiki" in names
    assert sel.transform(X.to_numpy()).shape == (len(X), 3)


def test_stability_separates_signal_from_noise(report):
    table, _ = report
    assert table.loc["z_wiki", "stability_freq"] > 0.8 and table.loc["z_wiki", "stable"]
    assert table.loc[[f"noise{i}" for i in range(5)], "stability_freq"].max() < 0.8


def test_lasso_path_keeps_strong_features_longest(report):
    table, _ = report
    strongest = max(table.loc["tone_neg", "lasso_survival_alpha"],
                    table.loc["tone_neg_copy", "lasso_survival_alpha"])
    assert strongest > table.loc["vix", "lasso_survival_alpha"]
    assert table.loc["vix", "lasso_survival_alpha"] > table.loc["noise0",
                                                                 "lasso_survival_alpha"]


def test_selection_happens_inside_each_fold(report):
    table, cv_scores = report
    assert set(cv_scores["method"]) == {"filter_mi", "filter_mrmr", "wrapper_rfe",
                                        "embedded_lasso", "all_features"}
    assert (cv_scores["folds"] == 4).all()
    assert cv_scores.set_index("method").loc["filter_mrmr", "mean_n_kept"] == 3
    freq = table["filter_mi_fold_freq"]
    assert freq.between(0, 1).all() and freq.loc["z_wiki"] == 1.0


def test_selection_beats_noise_only_baseline(report):
    _, cv_scores = report
    s = cv_scores.set_index("method")["cv_mae"]
    assert s["filter_mrmr"] < 0.012  # y has sd ~0.02; selecting 3 good features helps


def test_families_are_labelled(report):
    table, _ = report
    assert table.loc["z_wiki", "family"] == "A_intensity"
    assert table.loc["noise0", "family"] == "other"


def test_main_runs_on_fixtures(monkeypatch):
    monkeypatch.setenv("KPR_USE_FIXTURES", "1")
    assert S.main([]) == 0
