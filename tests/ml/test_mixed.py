"""Owner: ml session. Tests for kpr.models.mixed (task M8, dossier §9.4, §10.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from kpr.models import mixed as M

PARAMS = {
    "mixed": {"controls": ["vix", "z_wiki", "empty_col"], "random_intercept": False,
              "min_events_per_person": 3},
    "scores": {"window_years": 3},
}


def _synthetic(n_persons: int = 30, seed: int = 0):
    """Persons with known slopes: bad news hurts about twice as much as good news helps."""
    rng = np.random.default_rng(seed)
    true = pd.DataFrame({
        "person_id": [f"p{i:02d}" for i in range(n_persons)],
        "slope_neg": rng.normal(-0.02, 0.008, n_persons),
        "slope_pos": rng.normal(0.01, 0.004, n_persons),
        "n": rng.integers(15, 60, n_persons),
    })
    true.loc[0, "n"] = 3  # one person with very few events
    rows = []
    for t in true.itertuples():
        tone = rng.normal(0, 1.5, t.n)
        neg, pos = np.maximum(0, -tone), np.maximum(0, tone)
        vix = rng.normal(20, 5, t.n)
        car = t.slope_neg * neg + t.slope_pos * pos + 0.0005 * (vix - 20)
        rows.append(pd.DataFrame({
            "person_id": t.person_id, "tone_neg": neg, "tone_pos": pos, "vix": vix,
            "z_wiki": rng.normal(5, 1, t.n), "empty_col": np.nan,
            "car_signed": car + rng.normal(0, 0.01, t.n),
            "j0": pd.to_datetime(rng.choice(pd.bdate_range("2016-01-01", "2018-12-31"), t.n)),
        }))
    return pd.concat(rows, ignore_index=True), true.set_index("person_id")


@pytest.fixture(scope="module")
def fitted():
    frame, true = _synthetic()
    data = M.prepare(frame, PARAMS["mixed"]["controls"], 3)
    result = M.fit_mixed(data)
    return frame, true, data, result, M.person_slopes(result, data)


def test_prepare_standardises_controls_and_skips_empty_ones():
    frame, _ = _synthetic()
    frame.loc[frame.index[:2], "vix"] = np.nan
    data = M.prepare(frame, ["vix", "z_wiki", "empty_col", "missing_col"], 3)
    assert data.attrs["controls"] == ["vix", "z_wiki"]
    assert data["vix"].notna().all()
    assert data["vix"].mean() == pytest.approx(0, abs=1e-9)
    assert data["vix"].std() == pytest.approx(1)


def test_prepare_drops_persons_with_too_few_events():
    frame, true = _synthetic()
    data = M.prepare(frame, [], min_events=10)
    assert "p00" not in set(data["person_id"])  # 3 events only
    assert data["person_id"].nunique() == len(true) - 1


def test_average_slopes_and_asymmetry(fitted):
    *_, slopes = fitted
    assert slopes.attrs["beta_neg"] == pytest.approx(-0.02, abs=0.005)
    assert slopes.attrs["beta_pos"] == pytest.approx(0.01, abs=0.004)
    assert abs(slopes.attrs["beta_neg"]) > abs(slopes.attrs["beta_pos"])  # H2 direction


def test_person_slopes_recover_the_ranking(fitted):
    _, true, _, _, slopes = fitted
    est = slopes.set_index("person_id").loc[true.index]
    assert spearmanr(est["slope_neg"], true["slope_neg"]).statistic > 0.7
    assert np.allclose(est["slope_neg"], slopes.attrs["beta_neg"] + est["b_neg"])
    assert est.loc["p00", "n_events"] == 3


def test_few_events_are_shrunk_towards_the_average(fitted):
    frame, _, _, _, slopes = fitted
    est = slopes.set_index("person_id")
    few = frame[frame["person_id"] == "p00"]
    ols = np.linalg.lstsq(few[["tone_neg", "tone_pos"]].to_numpy(),
                          few["car_signed"].to_numpy(), rcond=None)[0]
    beta = slopes.attrs["beta_neg"]
    assert abs(est.loc["p00", "slope_neg"] - beta) < abs(ols[0] - beta)


def test_rolling_windows_give_one_slope_per_person_and_year():
    frame, true = _synthetic()
    out = M.rolling_slopes(frame, [2018, 2019], PARAMS)
    assert set(out["year"]) == {2018, 2019}
    assert out.groupby("year")["person_id"].nunique().loc[2018] == len(true)
    # the 2019 window (2017-2019) holds fewer events than the 2018 window (2016-2018)
    assert out.query("year == 2019")["n_events"].sum() < out.query("year == 2018")[
        "n_events"].sum()
    assert {"beta_neg", "beta_pos", "converged"} <= set(out.columns)


def test_main_runs_on_fixtures(monkeypatch):
    monkeypatch.setenv("KPR_USE_FIXTURES", "1")
    assert M.main([]) == 0
