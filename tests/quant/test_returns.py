"""Owner: quant session. Unit tests for kpr.build.returns (task Q2, dossier §7.5)."""

import numpy as np
import pandas as pd
import pytest

from kpr.build import returns as R

FIXTURES = R.REPO_ROOT / "tests" / "fixtures"


# ---- hand computation --------------------------------------------------------------------------
#
# Event time k = -6 .. 3, J0 at k = 0. Estimation window [-6, -2] (5 days), gap at k = -1.
#
#   k     -6      -5      -4      -3     -2      -1     0      1      2      3
#   R_m   0.010  -0.020   0.015   0.000  -0.005  0.03   0.020 -0.010  0.005  0.0
#   R     0.012  -0.028   0.020   0.002  -0.004  0.50   0.035 -0.010  0.010  0.0
#
# mean(R_m) = 0, mean(R) = 0.002 / 5 = 0.0004
# Sxy = 0.01*0.012 + 0.02*0.028 + 0.015*0.02 + 0 + 0.005*0.004 = 0.001
# Sxx = 0.0001 + 0.0004 + 0.000225 + 0 + 0.000025            = 0.00075
# beta = Sxy / Sxx = 4/3, alpha = 0.0004 - beta * 0 = 0.0004
# AR_0 = 0.035 - (0.0004 + 4/3 * 0.020) =  0.0079333...
# AR_1 = -0.010 - (0.0004 - 4/3 * 0.010) =  0.0029333...
# AR_2 = 0.010 - (0.0004 + 4/3 * 0.005) =  0.0029333...
# CAR[0, 2] = 0.035 - 3 * 0.0004 - 4/3 * 0.015 = 0.0138
# Market-adjusted CAR[0, 2] = 0.035 - 0.015 = 0.020
# Residuals: -0.0052/3, -0.0052/3, -0.0004, 0.0016, 0.0068/3 -> SSE = 1.3866667e-5
# sigma = sqrt(SSE / (5 - 2)) = 0.00214994...

DATES = pd.bdate_range("2024-01-01", periods=10)
RM = pd.Series([0.010, -0.020, 0.015, 0.000, -0.005, 0.03, 0.020, -0.010, 0.005, 0.0], DATES)
RS = pd.Series([0.012, -0.028, 0.020, 0.002, -0.004, 0.50, 0.035, -0.010, 0.010, 0.0], DATES)
J0 = DATES[6]


def _hand(model="market"):
    return R.event_abnormal_returns(RS, RM, J0, model=model, estimation_window=(-6, -2),
                                    event_window=(-6, 3), min_obs=5)


def test_market_model_matches_hand_computation():
    res = _hand()
    assert res.fit.params["beta"] == pytest.approx(4 / 3, abs=1e-12)
    assert res.fit.params["alpha"] == pytest.approx(0.0004, abs=1e-12)
    assert res.fit.n_obs == 5
    assert res.fit.sigma == pytest.approx(np.sqrt(1.3866667e-5 / 3), rel=1e-6)
    assert res.frame.loc[0, "ar"] == pytest.approx(0.035 - 0.0004 - 4 / 3 * 0.020, abs=1e-12)
    assert res.frame.loc[1, "ar"] == pytest.approx(0.0029333333, abs=1e-9)
    assert res.car(0, 2) == pytest.approx(0.0138, abs=1e-12)


def test_estimation_window_ignores_the_gap_day():
    # The 50 % return at k = -1 would change beta if it leaked into the estimation window.
    res = _hand()
    assert res.frame.loc[-1, "ar"] == pytest.approx(0.50 - 0.0004 - 4 / 3 * 0.03)
    assert res.fit.params["beta"] == pytest.approx(4 / 3)


def test_market_adjusted_matches_hand_computation():
    res = _hand("market_adjusted")
    assert res.car(0, 2) == pytest.approx(0.020, abs=1e-12)
    assert res.car(0, 0) == pytest.approx(0.015, abs=1e-12)


def test_car_windows_and_missing_values():
    res = _hand()
    assert res.car(0, 1) == pytest.approx(res.frame.loc[0:1, "ar"].sum())
    assert res.car(-1, 1) == pytest.approx(res.frame.loc[-1:1, "ar"].sum())
    assert np.isnan(res.car(0, 5))  # runs past the data
    with pytest.raises(ValueError):
        res.car(2, 0)
    gappy = RS.copy()
    gappy.iloc[7] = np.nan  # k = 1
    res2 = R.event_abnormal_returns(gappy, RM, J0, estimation_window=(-6, -2),
                                    event_window=(0, 2), min_obs=5)
    assert np.isnan(res2.car(0, 2))
    assert not np.isnan(res2.car(0, 0))


def test_too_few_estimation_days_gives_no_fit():
    res = R.event_abnormal_returns(RS, RM, J0, estimation_window=(-6, -2), event_window=(0, 2),
                                   min_obs=6)
    assert not res.fit.ok and res.fit.n_obs == 5
    assert np.isnan(res.car(0, 2))


def test_sessions_define_event_time_not_price_rows():
    # A session with no price row stays in event time as a missing return.
    sessions = DATES
    rs = RS.drop(DATES[8])  # k = 2 missing from the price table
    res = R.event_abnormal_returns(rs, RM, J0, sessions=sessions, estimation_window=(-6, -2),
                                   event_window=(0, 3), min_obs=5)
    assert res.frame.loc[2, "date"] == DATES[8]
    assert np.isnan(res.frame.loc[2, "ar"])
    assert np.isnan(res.car(0, 2))
    assert not np.isnan(res.car(0, 1))


def test_j0_must_be_a_session():
    with pytest.raises(ValueError, match="not one of the given sessions"):
        R.event_abnormal_returns(RS, RM, "2024-01-06", min_obs=5)


def test_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        R.event_abnormal_returns(RS, RM, J0, model="capm2")


# ---- Fama-French 3 -----------------------------------------------------------------------------

def test_ff3_recovers_exact_loadings():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2023-01-02", periods=80)
    f = pd.DataFrame(rng.normal(0, 0.01, (80, 3)), index=days, columns=["mkt_rf", "smb", "hml"])
    f["rf"] = 0.0001
    ret = f["rf"] + 0.0002 + 1.1 * f["mkt_rf"] + 0.4 * f["smb"] - 0.3 * f["hml"]
    ret.iloc[75] += 0.05  # abnormal move at J0
    res = R.event_abnormal_returns(ret, f["mkt_rf"] + f["rf"], days[75], model="ff3",
                                   estimation_window=(-70, -10), event_window=(0, 2),
                                   factors=f, min_obs=30)
    p = res.fit.params
    assert p["b"] == pytest.approx(1.1) and p["s"] == pytest.approx(0.4)
    assert p["h"] == pytest.approx(-0.3) and p["a"] == pytest.approx(0.0002)
    assert res.car(0, 2) == pytest.approx(0.05)


def test_ff3_needs_factors():
    with pytest.raises(ValueError, match="needs `factors`"):
        R.event_abnormal_returns(RS, RM, J0, model="ff3", min_obs=5)


def test_read_ff3_csv(tmp_path):
    path = tmp_path / "F-F_Research_Data_Factors_daily.CSV"
    path.write_text(
        "This file was created by CMPT_ME_BEME_RETS_DAILY using the 202409 CRSP database.\n"
        "The Tbill return is the simple daily rate.\n\n"
        ",Mkt-RF,SMB,HML,RF\n"
        "19260701,    0.10,   -0.25,   -0.27,    0.009\n"
        "19260702,    0.45,   -0.33,   -0.06,    0.009\n\n"
        "Copyright 2024 Kenneth R. French\n",
        encoding="latin-1",
    )
    df = R.read_ff3_csv(path)
    assert list(df.columns) == ["mkt_rf", "smb", "hml", "rf"]
    assert df.index[0] == pd.Timestamp("1926-07-01")
    assert df.loc["1926-07-02", "mkt_rf"] == pytest.approx(0.0045)
    assert len(df) == 2


# ---- many events, on the synthetic fixtures ------------------------------------------------

@pytest.fixture(scope="module")
def fixtures():
    return {n: pd.read_parquet(FIXTURES / f"{n}.parquet") for n in ("prices", "market", "events")}


def test_compute_cars_recovers_planted_shocks(fixtures):
    ev, prices, market = fixtures["events"], fixtures["prices"], fixtures["market"]
    mapping = {"FAKE1": "^GSPC", "FAKE2": "^GSPC", "FAKE3": "^STOXX"}
    cars = R.compute_cars(ev, prices, market, mapping, min_obs=120)
    assert list(cars.columns) == ["event_id", "car_0_2", "car_m5_m1", "car_0_1", "car_0_5",
                                  "alpha", "beta", "sigma", "n_est"]
    assert len(cars) == len(ev)
    m = ev.merge(cars, on="event_id", suffixes=("_true", ""))
    full = m[m.n_est == 221]
    assert len(full) >= 10
    # Fixture betas: FAKE1 1.3, FAKE2 0.9, FAKE3 1.0 (daily noise 1.2 %, 221 days).
    betas = full.groupby("company_id")["beta"].mean()
    for cid, beta in {"FAKE1": 1.3, "FAKE2": 0.9, "FAKE3": 1.0}.items():
        if cid in betas:
            assert betas[cid] == pytest.approx(beta, abs=0.25)
    # The planted 3-day moves are recovered: the noise of a 3-day CAR is about 2 % and many
    # planted moves are under 1 %, so we check the correlation and the slope, not each event.
    assert full["car_0_2"].corr(full["car_0_2_true"]) > 0.5
    slope = np.polyfit(full["car_0_2_true"], full["car_0_2"], 1)[0]
    assert 0.6 < slope < 1.4


def test_compute_cars_short_history_still_gives_car_with_n_est(fixtures):
    ev, prices, market = fixtures["events"], fixtures["prices"], fixtures["market"]
    mapping = {"FAKE1": "^GSPC", "FAKE2": "^GSPC", "FAKE3": "^STOXX"}
    cars = R.compute_cars(ev, prices, market, mapping)
    short = cars[(cars.n_est >= R.MIN_FIT_OBS) & (cars.n_est < 120)]
    assert short["car_0_2"].notna().all()  # events.py excludes them, but keeps a CAR
    none = cars[cars.n_est < R.MIN_FIT_OBS]
    assert none["car_0_2"].isna().all()


def test_compute_cars_unknown_company_gives_missing_row(fixtures):
    ev = fixtures["events"].head(2).assign(company_id=["NOPE", "FAKE1"])
    cars = R.compute_cars(ev, fixtures["prices"], fixtures["market"], {"FAKE1": "^GSPC"},
                          model="market_adjusted")
    assert np.isnan(cars.loc[0, "car_0_2"]) and cars.loc[0, "n_est"] == 0
    assert len(cars) == 2


def test_params_match_the_dossier():
    p = R.load_params()
    assert p["estimation_window"] == [-250, -30]
    assert p["car_window"] == [0, 2]


def test_main_runs_on_fixtures(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("KPR_USE_FIXTURES", "1")
    out = tmp_path / "cars.csv"
    assert R.main(["--out", str(out)]) == 0
    assert out.exists()
    assert "car_0_2" in capsys.readouterr().out
