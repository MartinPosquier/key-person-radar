"""Owner: quant session. Tests of kpr.validation.placebo (task Q8, dossier §11.3)."""

import numpy as np
import pandas as pd
import pytest

from kpr.build import returns as R
from kpr.validation import placebo as P

FIXTURES = P.REPO_ROOT / "tests" / "fixtures"
SECTORS = {"FAKE1": "Software", "FAKE2": "Software", "FAKE3": "Banks"}


@pytest.fixture(scope="module")
def fx():
    return {n: pd.read_parquet(FIXTURES / f"{n}.parquet")
            for n in ("events", "panel", "prices", "market")}


@pytest.fixture(scope="module")
def placebos(fx):
    return P.run_placebos(fx["events"], fx["panel"], fx["prices"], fx["market"], universe={},
                          sectors=SECTORS)


def test_calm_days_avoid_shocks(fx):
    calm = P.calm_days(fx["panel"], fx["events"])
    assert (calm["date"].notna()).all() and len(calm) > 100
    for pid, dates in fx["events"].groupby("person_id")["date_utc"]:
        d = calm.loc[calm.person_id == pid, "date"]
        gaps = np.abs((d.to_numpy()[:, None] - dates.to_numpy()[None, :]) / np.timedelta64(1, "D"))
        assert (gaps > P.CALM_DISTANCE_DAYS).all()


def test_random_dates_match_counts_and_signs(fx):
    base = P.ES.select(fx["events"])
    rd = P.random_dates(base, fx["panel"], np.random.default_rng(0), all_shocks=fx["events"])
    assert rd.groupby("person_id").size().to_dict() == base.groupby("person_id").size().to_dict()
    for pid, g in rd.groupby("person_id"):
        assert sorted(g["sign"]) == sorted(base.loc[base.person_id == pid, "sign"])
    calm = P.calm_days(fx["panel"], fx["events"])
    keys = set(zip(calm.person_id, calm.date, strict=True))
    assert all(k in keys for k in zip(rd.person_id, rd.date_utc, strict=True))


def test_random_dates_are_reproducible(fx):
    base = P.ES.select(fx["events"])
    a = P.random_dates(base, fx["panel"], np.random.default_rng(1))
    b = P.random_dates(base, fx["panel"], np.random.default_rng(1))
    pd.testing.assert_frame_equal(a, b)


def test_wrong_company_changes_sector():
    ev = pd.DataFrame({"event_id": ["e1", "e2"], "person_id": "p",
                       "company_id": ["FAKE1", "FAKE3"],
                       "date_utc": pd.to_datetime(["2021-03-01", "2021-04-01"]),
                       "sign": ["negative", "positive"]})
    out = P.wrong_company(ev, ["FAKE1", "FAKE2", "FAKE3"], SECTORS, np.random.default_rng(0))
    assert list(out["company_id"]) == ["FAKE3", "FAKE1"] or list(out["company_id"]) == [
        "FAKE3", "FAKE2"]
    assert all(SECTORS[a] != SECTORS[b] for a, b in zip(ev.company_id, out.company_id,
                                                        strict=True))
    # Unknown sectors: any other company.
    any_other = P.wrong_company(ev, ["FAKE1", "FAKE2", "FAKE3"], {}, np.random.default_rng(0))
    assert (any_other["company_id"] != ev["company_id"]).all()


def test_time_reversal_is_car_minus10_minus8(fx, placebos):
    tr = placebos[placebos.placebo == "time_reversal"].dropna(subset=["car_0_2"]).iloc[0]
    prices, market = fx["prices"], fx["market"]
    stock = prices[prices.company_id == tr.company_id].set_index("date")["ret"]
    index_id = {"FAKE1": "^GSPC", "FAKE2": "^GSPC", "FAKE3": "^STOXX"}[tr.company_id]
    mkt = market[market.index_id == index_id].set_index("date")["ret"]
    res = R.event_abnormal_returns(stock, mkt, tr.j0, event_window=(-10, 10))
    assert tr.car_0_2 == pytest.approx(res.car(-10, -8))


def test_run_placebos_shape(placebos):
    assert set(placebos["placebo"]) == {"random_dates", "wrong_company", "time_reversal"}
    assert list(placebos.columns) == P.OUT_COLUMNS
    assert placebos.groupby("placebo")["event_id"].apply(lambda s: s.is_unique).all()
    short = placebos[placebos["n_est"] < 120]
    assert short["car_0_2"].isna().all()


def test_placebos_show_no_effect_where_real_events_do(fx, placebos):
    table = P.summary(placebos, fx["events"])
    assert table.loc[("real", "negative"), "p_t"] < 0.01
    for name in ("random_dates", "wrong_company", "time_reversal"):
        assert table.loc[(name, "negative"), "p_t"] > 0.05, name
        assert abs(table.loc[(name, "negative"), "mean"]) < abs(
            table.loc[("real", "negative"), "mean"])


def test_unrelated_people(fx):
    celeb = fx["panel"][fx["panel"].person_id == "p_alpha"].assign(person_id="celeb_1")
    out = P.unrelated_people(celeb, ["FAKE2", "FAKE3"], np.random.default_rng(0),
                             P.E.load_params())
    assert len(out) >= 5 and set(out["company_id"]) <= {"FAKE2", "FAKE3"}
    assert out["event_id"].str.endswith("-up").all()
    full = P.run_placebos(fx["events"], fx["panel"], fx["prices"], fx["market"],
                          unrelated_panel=celeb, universe={}, sectors=SECTORS)
    assert "unrelated_people" in set(full["placebo"])


def test_main_writes_only_with_out_dir(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("KPR_USE_FIXTURES", "1")
    assert P.main([]) == 0
    assert "random_dates" in capsys.readouterr().out
    assert P.main(["--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "placebo_events.parquet").exists()
    assert (tmp_path / "placebo_summary.csv").exists()
