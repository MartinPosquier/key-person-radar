"""Owner: ml session. Tests for kpr.models.explain (task M11, dossier §9.8, §9.9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kpr.models import explain as EX
from kpr.models.cv import modelling_frame
from kpr.models.tune import make_model


def _synthetic(n: int = 1200, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(rng.choice(pd.bdate_range("2015-01-01", "2023-12-31"), n))
    tone = rng.normal(0, 1, n)
    ceo = rng.integers(0, 2, n).astype(float)
    feats = pd.DataFrame({
        "event_id": [f"e{i}" for i in range(n)], "tone_rel": tone,
        "tone_neg": np.maximum(0, -tone), "tone_pos": np.maximum(0, tone),
        "attention_share": rng.uniform(0, 1, n), "vix": rng.normal(20, 5, n),
        "mkt_ret_j0": np.nan,  # empty column, as with an empty universe
        "role_CEO": ceo, "role_CFO": 1 - ceo,
    })
    car = -0.03 * feats["tone_neg"] + 0.01 * feats["tone_pos"] + rng.normal(0, 0.004, n)
    events = pd.DataFrame({
        "event_id": feats["event_id"], "person_id": rng.choice(["pa", "pb"], n),
        "company_id": "C", "j0": dates, "car_0_2": car, "excluded": False,
        "is_natural_experiment": False, "tone_rel": tone, "z_wiki": 5.0,
        "sign": np.where(tone < -1, "negative", np.where(tone > 1, "positive", "neutral")),
        "split": np.where(dates.year <= 2021, "train", "validation"),
    })
    return events, feats.drop(columns=[]), [c for c in feats.columns if c != "event_id"]


@pytest.fixture(scope="module")
def setup():
    events, feats, cols = _synthetic()
    train = modelling_frame(events, feats, ("train",))
    pipes = {"tree": make_model("hgb", cols).fit(train[cols], train["car_signed"]),
             "lasso": make_model("lasso", cols).set_params(model__alpha=1e-4)
             .fit(train[cols], train["car_signed"])}
    out = EX.explain(events, feats, {}, pipes, event_id=events.query(
        "split == 'validation'")["event_id"].iloc[0])
    return events, feats, cols, pipes, out


def test_shap_values_add_up_to_the_prediction(setup):
    events, feats, cols, pipes, out = setup
    values = out["shap_values"]
    assert "mkt_ret_j0" not in values.columns  # empty column dropped inside the fold
    valid = modelling_frame(events, feats, ("validation",))
    pred = pipes["tree"].predict(valid[cols])
    assert np.allclose(values.sum(axis=1) + values.attrs["expected_value"], pred, atol=1e-6)


def test_shap_summary_ranks_the_driving_feature_first(setup):
    *_, out = setup
    summary = out["shap_summary"]
    assert summary.index[0] in ("tone_neg", "tone_rel")
    assert summary["share"].sum() == pytest.approx(1.0)
    assert summary.loc["tone_neg", "family"] == "B_tone"
    assert out["family_importance"].index[0] == "B_tone"


def test_lasso_coefficients_in_car_points(setup):
    *_, out = setup
    coef = out["lasso_coefficients"]
    assert coef["tone_neg"] < 0  # bad news lowers the CAR
    assert abs(coef.iloc[0]) >= abs(coef.iloc[-1])


def test_local_explanation_is_sorted(setup):
    *_, out = setup
    local = out["local"]
    assert list(local["shap"].abs()) == sorted(local["shap"].abs(), reverse=True)
    assert {"value", "shap"} <= set(local.columns)


def test_error_analysis(setup):
    *_, out = setup
    largest, by_group = out["largest_errors"], out["errors_by_group"]
    assert len(largest) == EX.N_LARGEST
    assert (largest["error"].abs().diff().dropna() <= 1e-12).all()
    assert set(by_group["dimension"]) == {"sector", "region", "role", "sign", "year"}
    roles = by_group.query("dimension == 'role'")
    assert set(roles["group"]) == {"CEO", "CFO"}
    assert set(by_group.query("dimension == 'sector'")["group"]) == {"unknown"}
    assert "review_note" in largest.columns


def test_error_by_group_by_hand():
    frame = pd.DataFrame({"event_id": list("abcd"), "j0": pd.to_datetime(["2022-01-03"] * 4),
                          "car_signed": [0.0, 0.0, 0.0, 0.0], "sign": ["negative"] * 2 + [
                              "positive"] * 2, "role_CEO": [1.0, 1, 0, 0],
                          "role_CFO": [0.0, 0, 1, 1]})
    by_group, largest = EX.error_analysis(frame, np.array([0.01, 0.03, 0.0, 0.02]))
    ceo = by_group.query("dimension == 'role' and group == 'CEO'")["mae"].iloc[0]
    assert ceo == pytest.approx(0.02)
    assert largest["event_id"].iloc[0] == "b"


def test_plots_are_written(setup, tmp_path):
    events, feats, *_, out = setup
    EX.save_plots(out["shap_values"], modelling_frame(events, feats, ("validation",)), tmp_path)
    assert (tmp_path / "shap_summary.png").stat().st_size > 1000
    assert (tmp_path / "shap_dependence_tone.png").exists()


def test_main_runs_on_fixtures(monkeypatch):
    monkeypatch.setenv("KPR_USE_FIXTURES", "1")
    assert EX.main([]) == 0
