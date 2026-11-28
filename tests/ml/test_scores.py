"""Owner: ml session. Tests for kpr.scores (task M10, dossier §3, §10)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from kpr.schemas import validate
from kpr.scores import sensitivity as SE
from kpr.scores import smoothing as SM
from kpr.scores import value as VA

PARAMS = {"standard_shock": {"z_wiki": 4.0},
          "scores": {"window_years": 3, "shrinkage_k": 10, "smoothing_omega": 0.4,
                     "max_yearly_change": 2.0, "agreement_min_spearman": 0.6}}
FEATS = ["z_wiki", "tone_rel", "tone_neg", "tone_pos", "founder", "tone_x_founder",
         "attention_share", "tone_x_attention", "z_company", "specificity"]


class LinearCar:
    """Deterministic 'model': CAR = -0.01 tone- + 0.005 tone+ - 0.001 z_wiki."""

    def predict(self, X):
        return -0.01 * X["tone_neg"] + 0.005 * X["tone_pos"] - 0.001 * X["z_wiki"]


def _frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p, cid in (("pa", "C1"), ("pb", "C1"), ("pc", "C2"), ("pd", "C3")):
        n = 30
        tone = rng.normal(0, 1, n)
        rows.append(pd.DataFrame({
            "event_id": [f"{p}{i}" for i in range(n)], "person_id": p, "company_id": cid,
            "j0": pd.to_datetime(rng.choice(pd.bdate_range("2016-01-01", "2018-12-31"), n)),
            "z_wiki": rng.normal(5, 1, n), "tone_rel": tone, "tone_neg": np.maximum(0, -tone),
            "tone_pos": np.maximum(0, tone), "founder": float(p == "pa"),
            "attention_share": 0.3, "z_company": rng.normal(1, 1, n),
        }))
    f = pd.concat(rows, ignore_index=True)
    f["tone_x_founder"] = f["tone_rel"] * f["founder"]
    f["tone_x_attention"] = f["tone_rel"] * f["attention_share"]
    f["specificity"] = f["z_wiki"] - f["z_company"]
    f["car_0_2"] = LinearCar().predict(f) + rng.normal(0, 0.002, len(f))
    return f


# ---- method A -------------------------------------------------------------------------------


def test_window_profiles_and_standard_shocks():
    f = _frame()
    prof = SE.window_profiles(f, FEATS, [2018], window=3)
    pa = prof.set_index("person_id").loc["pa"]
    assert pa["n_events"] == 30
    assert pa["tone_q10"] == pytest.approx(f.query("person_id == 'pa'")["tone_rel"].quantile(.1))
    neg, pos = SE.standard_shocks(prof, FEATS, PARAMS)
    assert (neg["z_wiki"] == 4).all() and (neg["tone_pos"] == 0).all()
    assert np.allclose(neg["tone_neg"], -prof["tone_q10"])
    assert np.allclose(neg["tone_x_founder"], prof["tone_q10"] * prof["founder"])
    assert np.allclose(neg["specificity"], prof["specificity"])  # kept at its mean (§10.1)


def test_method_a_exact_values():
    f = _frame()
    prof = SE.window_profiles(f, FEATS, [2018], 3)
    a = SE.method_a(LinearCar(), prof, FEATS, PARAMS).set_index("person_id")
    q10 = prof.set_index("person_id")["tone_q10"]
    expected = 100 * np.abs(-0.01 * -q10 - 0.004)
    assert np.allclose(a["a_minus"], expected.loc[a.index])


def test_method_a_bootstrap_interval():
    f = _frame()
    prof = SE.window_profiles(f, FEATS, [2018], 3)
    model = LinearRegression().fit(f[FEATS], f["car_0_2"])
    a = SE.method_a(model, prof, FEATS, PARAMS, f[FEATS], f["car_0_2"], n_boot=30)
    assert (a["a_minus_lo"] <= a["a_minus"]).all() and (a["a_minus"] <= a["a_minus_hi"]).all()


# ---- method B and combination -------------------------------------------------------------------


def test_method_b_exact_values_and_interval_through_zero():
    prof = pd.DataFrame({"person_id": ["pa", "pb"], "company_id": "C1", "year": 2018,
                         "tone_q10": [-2.0, -1.0], "tone_q90": [1.0, 2.0]})
    slopes = pd.DataFrame({"person_id": ["pa", "pb"], "year": 2018, "slope_neg": [-0.02, -0.01],
                           "slope_pos": [0.01, 0.001], "se_neg": [0.001, 0.001],
                           "se_pos": [0.001, 0.01]})
    b = SE.method_b(slopes, prof).set_index("person_id")
    assert b.loc["pa", "b_minus"] == pytest.approx(100 * 0.02 * 2)
    assert b.loc["pb", "b_plus"] == pytest.approx(100 * 0.001 * 2)
    assert b.loc["pb", "b_plus_lo"] == 0.0  # interval crosses zero
    assert b.loc["pa", "b_minus_lo"] < b.loc["pa", "b_minus"] < b.loc["pa", "b_minus_hi"]


def _ab(b_minus):
    key = {"person_id": list("abcde"), "company_id": "C", "year": 2020}
    a = pd.DataFrame({**key, "n_events": 10, "a_minus": [1., 2, 3, 4, 5],
                      "a_plus": [1., 2, 3, 4, 5]})
    b = pd.DataFrame({**key, "b_minus": b_minus, "b_plus": b_minus,
                      **{f"b_{s}_{x}": b_minus for s in ("minus", "plus") for x in ("lo", "hi")}})
    return a, b


def test_combine_mean_when_methods_agree_else_method_b():
    a, b = _ab([2., 3, 4, 5, 6])  # same ranking
    agree = SE.combine(a, b, 0.6)
    assert (agree["method"] == "A+B").all()
    assert agree["method_agreement"].iloc[0] == pytest.approx(1.0)
    assert np.allclose(agree["s_minus_raw"], [1.5, 2.5, 3.5, 4.5, 5.5])
    a, b = _ab([6., 5, 4, 3, 2])  # reversed ranking
    disagree = SE.combine(a, b, 0.6)
    assert (disagree["method"] == "B").all()
    assert np.allclose(disagree["s_minus_raw"], [6, 5, 4, 3, 2])


# ---- scale, shrinkage, smoothing ----------------------------------------------------------------


def test_scale_maps_deciles_and_caps_at_ten():
    thresholds = list(np.linspace(0, 5, 11))
    assert np.allclose(SM.to_scale([0, 2.5, 5, 50, -1], thresholds), [0, 5, 10, 10, 0])


def test_scale_is_computed_once_then_frozen(tmp_path):
    path = tmp_path / "score_scale.yaml"
    path.write_text("computed: false\n", encoding="utf-8")
    sens = pd.DataFrame({"year": [2016] * 11 + [2024], "s_minus_raw": [*range(11), 99.0],
                         "s_plus_raw": [*range(11), 99.0]})
    first = SM.load_or_compute_scale(sens, path=path)
    assert first["s_minus_thresholds"] == pytest.approx(list(range(11)))  # 2024 ignored
    again = SM.load_or_compute_scale(sens.assign(s_minus_raw=sens["s_minus_raw"] * 3),
                                     path=path)
    assert again["s_minus_thresholds"] == first["s_minus_thresholds"]


def test_shrinkage_weights():
    v = pd.Series([10.0, 0.0])
    groups, years = pd.Series(["g", "g"]), pd.Series([2020, 2020])
    out = SM.shrink(v, pd.Series([0, 1e9]), groups, years, k=10)
    assert out.iloc[0] == pytest.approx(5.0)  # no event: the group mean
    assert out.iloc[1] == pytest.approx(0.0, abs=1e-6)  # many events: its own value


def test_smoothing_caps_changes_and_resets_at_succession():
    rows = pd.DataFrame({"person_id": "p", "company_id": "C", "year": [2018, 2019, 2020],
                         "target": [2.0, 10.0, 10.0], "group_mean": 5.0,
                         "succession_flag": [False, False, False]})
    out = SM.smooth(rows, "target", omega=0.4, cap=2.0).tolist()
    assert out[0] == 2.0
    assert out[1] == pytest.approx(4.0)  # 0.4*10 + 0.6*2 = 5.2, capped at 2 + 2
    assert out[2] == pytest.approx(min(0.4 * 10 + 0.6 * 4.0, 6.0))
    new = pd.DataFrame({"person_id": "q", "company_id": "C", "year": [2020],
                        "target": [10.0], "group_mean": 2.0, "succession_flag": [True]})
    assert SM.smooth(new, "target", 0.4, 2.0).iloc[0] == pytest.approx(0.4 * 10 + 0.6 * 2)


def test_succession_flags_from_mandates():
    rows = pd.DataFrame({"person_id": ["old", "new", "new"], "company_id": "C",
                         "year": [2018, 2019, 2020]})
    mandates = pd.DataFrame({"person_id": ["old", "new"], "company_id": "C",
                             "start": pd.to_datetime(["2005-01-01", "2019-03-01"])})
    assert SM.succession_flags(rows, mandates).tolist() == [False, True, False]


# ---- value ----------------------------------------------------------------------------------


def test_calibration_uses_unwanted_training_departures_only():
    events = pd.DataFrame({"event_id": ["e1", "e2", "e3", "e4"], "company_id": ["A", "A", "B", "C"],
                           "j0": pd.to_datetime(["2018-05-02", "2018-05-04", "2019-01-10",
                                                 "2023-03-01"]),
                           "z_wiki": [4.0, 6.0, 5.0, 5.0], "car_0_2": [-0.05, -0.08, 0.01, -0.1]})
    deps = pd.DataFrame({"company_id": ["A", "B", "C"],
                         "date": pd.to_datetime(["2018-05-01", "2019-01-09", "2023-03-01"]),
                         "wanted_by_company": [False, True, False]})
    calib = VA.calibration_events(deps, events, pd.Timestamp("2021-12-31"))
    assert calib["event_id"].tolist() == ["e2"]  # largest spike of A; B wanted; C too late


def test_lambda_and_v():
    f = _frame()
    prof = SE.window_profiles(f, FEATS, [2018], 3)
    calib = f.iloc[:4].assign(car_0_2=[-0.06, -0.04, -0.05, -0.05])
    pred = np.abs(LinearCar().predict(f.iloc[:4][FEATS]))
    out = VA.departure_value(LinearCar(), prof, calib, f[["event_id", *FEATS]], FEATS)
    lam = 0.05 / pred.mean()
    assert out["lambda"].iloc[0] == pytest.approx(lam)
    shock = SE.shocked(prof, FEATS, calib["z_wiki"].mean(), calib["tone_rel"].mean(),
                       calib["specificity"].mean())
    assert np.allclose(out["v_pct"], 100 * lam * np.abs(LinearCar().predict(shock)))


def test_pi_share_by_hand():
    dates = pd.bdate_range("2020-01-01", periods=10)
    prices = pd.DataFrame({"company_id": "C", "date": dates, "ret": 0.01})
    prices.loc[4, "ret"] = 0.05  # shock day
    market = pd.DataFrame({"index_id": "^X", "date": dates, "ret": 0.0})
    events = pd.DataFrame({"person_id": ["p"], "company_id": ["C"], "j0": [dates[4]]})
    companies = pd.DataFrame({"company_id": ["C"], "market_index": ["^X"]})
    pi = VA.pi_share(events, prices, market, companies).iloc[0]
    assert pi["pi_share"] == pytest.approx((0.05**2 + 2 * 0.01**2) / (0.05**2 + 9 * 0.01**2))


# ---- full table -----------------------------------------------------------------------------


def test_build_scores_gives_a_valid_table():
    key = {"person_id": ["pa", "pa", "pb"], "company_id": "C1", "year": [2019, 2020, 2020]}
    sens = pd.DataFrame({**key, "n_events": [12, 5, 9], "method_agreement": 0.7,
                         **{f"s_{s}_raw{x}": [1.0, 2.0, 3.0] for s in ("minus", "plus")
                            for x in ("", "_lo", "_hi")}})
    value = pd.DataFrame({**key, "v_pct": [1.0, 2.0, 3.0], "v_lo": 0.5, "v_hi": 4.0,
                          "v_usd": 1e9, "pi_share": 0.1})
    mandates = pd.DataFrame({"person_id": ["pa", "pb"], "company_id": "C1", "role": "CEO",
                             "start": pd.to_datetime(["2010-01-01", "2020-02-01"])})
    companies = pd.DataFrame({"company_id": ["C1"], "sector": ["Tech"]})
    scale = {"s_minus_thresholds": list(np.linspace(0, 4, 11)),
             "s_plus_thresholds": list(np.linspace(0, 4, 11))}
    scores = SM.build_scores(sens, value, mandates, companies, PARAMS, scale)
    validate(scores, "scores")
    assert ((scores["s_minus_lo"] <= scores["s_minus"])
            & (scores["s_minus"] <= scores["s_minus_hi"])).all()
    assert scores.set_index(["person_id", "year"]).loc[("pb", 2020), "succession_flag"]


def test_main_runs_on_fixtures(monkeypatch):
    monkeypatch.setenv("KPR_USE_FIXTURES", "1")
    assert SM.main([]) == 0
