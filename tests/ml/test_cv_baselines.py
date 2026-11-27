"""Owner: ml session. Tests for kpr.models.cv and kpr.models.baselines (task M3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from kpr.models import baselines as B
from kpr.models import cv


def _synthetic(n: int = 1400, seed: int = 0, driver: str = "vix") -> tuple[pd.DataFrame, ...]:
    """Events 2015-2021 whose CAR depends on one feature only."""
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(rng.choice(pd.bdate_range("2015-01-01", "2021-12-31"), n))
    feats = pd.DataFrame({
        "event_id": [f"e{i}" for i in range(n)],
        "vix": rng.normal(20, 5, n), "vol_20d": np.abs(rng.normal(0.02, 0.005, n)),
        "z_wiki": rng.normal(5, 1, n), "z_news": rng.normal(3, 1, n),
        "tone_rel": rng.normal(0, 1, n), "founder": rng.integers(0, 2, n).astype(float),
    })
    car = 0.004 * (feats[driver] - feats[driver].mean()) / feats[driver].std()
    car += rng.normal(0, 0.002, n)
    events = pd.DataFrame({
        "event_id": feats["event_id"], "company_id": rng.choice(["A", "B", "C"], n),
        "j0": dates, "car_0_2": car, "excluded": False, "is_natural_experiment": False,
        "split": "train",
    })
    return events, feats


# ---- splitter -------------------------------------------------------------------------------


def test_expanding_year_folds_with_gap():
    dates = pd.Series(pd.bdate_range("2015-01-01", "2021-12-31"))
    splitter = cv.ExpandingYearSplit(dates, first_train_end_year=2017, gap_sessions=10)
    folds = list(splitter.split(dates))
    assert splitter.get_n_splits() == len(folds) == 4  # validate 2018, 2019, 2020, 2021
    prev_train = 0
    for (train, val), year in zip(folds, range(2018, 2022), strict=True):
        assert (dates.iloc[val].dt.year == year).all()
        gap = pd.bdate_range(dates.iloc[train].max(), f"{year}-01-01", inclusive="right")
        assert len(gap) >= 10
        assert len(train) > prev_train  # expanding window
        prev_train = len(train)
        assert not set(train) & set(val)


def test_gap_drops_events_just_before_the_validation_year():
    dates = pd.Series(pd.to_datetime(["2017-06-01", "2017-12-27", "2018-03-01"]))
    (train, val), = cv.ExpandingYearSplit(dates, 2017, gap_sessions=10).split(dates)
    assert list(train) == [0] and list(val) == [2]


def test_splitter_plugs_into_sklearn():
    events, feats = _synthetic()
    splitter = cv.ExpandingYearSplit(events["j0"], 2017, 10)
    scores = cross_val_score(LinearRegression(), feats[["vix"]], events["car_0_2"],
                             cv=splitter)
    assert len(scores) == 4 and (scores > 0.5).all()


def test_splitter_rejects_misaligned_x():
    splitter = cv.ExpandingYearSplit(pd.bdate_range("2015-01-01", periods=10), 2015)
    with pytest.raises(ValueError, match="rows"):
        list(splitter.split(np.zeros((3, 1))))


# ---- modelling frame and targets -------------------------------------------------------------


def test_modelling_frame_filters_and_never_gives_test():
    events, feats = _synthetic(n=40)
    events.loc[0, "excluded"] = True
    events.loc[1, "is_natural_experiment"] = True
    events.loc[2, "split"] = "test"
    events.loc[3, "car_0_2"] = np.nan
    frame = cv.modelling_frame(events, feats)
    assert len(frame) == 36
    assert frame.attrs["drop_counts"] == {
        "events_in": 40, "dropped_other_split": 1, "dropped_excluded": 1,
        "dropped_natural_experiment": 1, "dropped_missing_car": 1,
        "dropped_missing_features": 0, "events_kept": 36,
    }
    assert frame["j0"].is_monotonic_increasing
    with pytest.raises(PermissionError):
        cv.modelling_frame(events, feats, ("train", "test"))
    assert len(cv.modelling_frame(events, feats, ("test",), allow_test=True)) == 1


def test_targets():
    events = pd.DataFrame({"event_id": ["a", "b", "c"], "car_0_2": [-0.10, 0.01, 0.02]})
    feats = pd.DataFrame({"event_id": ["a", "b", "c"], "vol_20d": [0.01, 0.01, np.nan]})
    out = cv.add_targets(events, feats).set_index("event_id")
    assert out.loc["a", "car_abs"] == pytest.approx(0.10)
    assert out.loc["a", "big_move"] == 1.0  # 0.10 > 2 * 0.01 * sqrt(3)
    assert out.loc["b", "big_move"] == 0.0
    assert np.isnan(out.loc["c", "big_move"])


def test_oof_predictions_fit_a_clone_per_fold():
    events, feats = _synthetic()
    model = LinearRegression()
    splitter = cv.ExpandingYearSplit(events["j0"], 2017, 10)
    oof = cv.oof_predictions(model, feats[["vix"]], events["car_0_2"], splitter,
                             ids=events["event_id"])
    assert oof["fold"].nunique() == 4
    assert set(oof["id"]) <= set(events["event_id"])
    with pytest.raises(NotFittedError):
        model.predict(feats[["vix"]])


# ---- baselines ------------------------------------------------------------------------------


def test_baselines_use_only_their_families():
    columns = ["vix", "vol_20d", "z_wiki", "z_news", "tone_rel", "founder"]
    models = B.make_baselines(columns)
    b1 = models["B1"].named_steps["prep"].transformers[0][2]
    b2 = models["B2"].named_steps["prep"].transformers[0][2]
    assert set(b1) == {"vix", "vol_20d"}
    assert set(b2) == {"vix", "vol_20d", "z_wiki", "z_news"}


def test_group_mean_regressor():
    X = pd.DataFrame({"company_id": ["A", "A", "B"]})
    model = B.GroupMeanRegressor().fit(X, [1.0, 3.0, 10.0])
    pred = model.predict(pd.DataFrame({"company_id": ["A", "B", "Z"]}))
    assert list(pred) == [2.0, 10.0, pytest.approx(14 / 3)]


@pytest.mark.parametrize("driver, winner, loser", [("vix", "B1", "B0"), ("z_wiki", "B2", "B1")])
def test_each_baseline_captures_its_own_family(driver, winner, loser):
    events, feats = _synthetic(driver=driver)
    params = {"cv": {"first_train_end_year": 2017, "gap_sessions": 10}}
    res = B.run(events, feats, params)
    signed = res[(res["target"] == "car_signed") & (res["stage"] == "cv")].set_index("model")
    assert signed.loc[winner, "mae"] < 0.8 * signed.loc[loser, "mae"]
    assert signed.loc[winner, "folds"] == 4
    assert set(res["target"]) == {"car_signed", "car_abs", "big_move"}


def test_main_runs_on_fixtures(monkeypatch):
    monkeypatch.setenv("KPR_USE_FIXTURES", "1")
    assert B.main([]) == 0
