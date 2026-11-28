"""Owner: quant session. Tests of kpr.validation.case_nvidia (task Q12, dossier §11.6).

The fixtures stand in for 2026: person p_alpha at FAKE1, year 2021, scores of 2020.
"""

import json

import numpy as np
import pandas as pd
import pytest

from kpr.build import returns as R
from kpr.validation import case_nvidia as C

FIXTURES = C.E.REPO_ROOT / "tests" / "fixtures"
CASE = {"person": "p_alpha", "company": "FAKE1", "peers": ["FAKE2", "FAKE3"], "year": 2021}


@pytest.fixture(scope="module")
def fx():
    names = ("wiki_views", "prices", "market", "scores", "earnings", "headlines", "events")
    return {n: pd.read_parquet(FIXTURES / f"{n}.parquet") for n in names}


@pytest.fixture(scope="module")
def case(fx):
    return C.run_case(fx["wiki_views"], fx["prices"], fx["market"], fx["scores"],
                      fx["earnings"], headlines=fx["headlines"], universe={}, **CASE)


def test_attention_peak_is_a_planted_shock(fx):
    peaks = C.find_attention_peaks(fx["wiki_views"], "p_alpha", 2021, top=3)
    planted = set(fx["events"].loc[fx["events"].person_id == "p_alpha", "date_utc"])
    assert peaks.index[0] in planted
    assert peaks["z_wiki"].is_monotonic_decreasing
    with pytest.raises(ValueError, match="no views"):
        C.find_attention_peaks(fx["wiki_views"], "nobody", 2021)


def test_case_cars_are_net_of_the_market(fx, case):
    car = case["cars"]["FAKE1"]
    day = pd.Timestamp(case["date"])
    stock = fx["prices"][fx["prices"].company_id == "FAKE1"].set_index("date")["ret"]
    mkt = fx["market"][fx["market"].index_id == "^GSPC"].set_index("date")["ret"]
    res = R.event_abnormal_returns(stock, mkt, day, event_window=(0, 5))
    assert car["car_0_2"] == pytest.approx(res.car(0, 2))
    assert car["car_0_5"] == pytest.approx(res.car(0, 5))
    assert car["raw_0_2"] == pytest.approx(stock.loc[day:].iloc[:3].sum())
    assert set(case["cars"]) == {"FAKE1", "FAKE2", "FAKE3"}


def test_competing_news(fx):
    d = pd.Timestamp("2020-08-04")  # FAKE1 releases on 2020-08-05
    news = C.competing_news(d, "FAKE1", "p_alpha", fx["earnings"], fx["headlines"])
    assert news["earnings_dates"] == ["2020-08-05"]
    planted = fx["headlines"].iloc[0]
    news = C.competing_news(planted.date, "FAKE1", planted.person_id, fx["earnings"],
                            fx["headlines"])
    assert planted.title in news["headlines"]


def test_model_expectation_uses_the_previous_year(fx):
    m = C.model_expectation(fx["scores"], "p_alpha", "FAKE1", 2020)
    assert m["expected_abs_car"] == pytest.approx(0.06) and m["s_minus"] == 9.0
    pos = C.model_expectation(fx["scores"], "p_alpha", "FAKE1", 2020, sign="positive")
    assert pos["expected_abs_car"] == pytest.approx(0.03)
    with pytest.raises(ValueError, match="no score"):
        C.model_expectation(fx["scores"], "p_alpha", "FAKE1", 2019)


@pytest.mark.parametrize("actual, word", [(-0.06, "consistent"), (-0.01, "overestimated"),
                                          (-0.20, "underestimated"), (np.nan, "not computable")])
def test_verdict(actual, word):
    assert C.verdict(actual, 0.06).startswith(word)


def test_run_case_and_report(case):
    assert case["date_source"].startswith("highest attention")
    assert case["model"]["year"] == 2020
    json.dumps(case)  # JSON-ready
    text = C.report(case)
    assert "| FAKE1 |" in text and "Verdict" in text and "illustration" in text


def test_hand_sourced_date_overrides_the_peak(fx):
    out = C.run_case(fx["wiki_views"], fx["prices"], fx["market"], fx["scores"],
                     fx["earnings"], universe={}, date="2021-06-14", **CASE)
    assert out["date"] == "2021-06-14" and out["date_source"].startswith("sourced by hand")


def test_main(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("KPR_USE_FIXTURES", "1")
    args = ["--person", "p_alpha", "--company", "FAKE1", "--peers", "FAKE2", "--year", "2021"]
    assert C.main([*args, "--out-dir", str(tmp_path)]) == 0
    assert "# Case study" in capsys.readouterr().out
    assert json.loads((tmp_path / "case_nvidia.json").read_text())["company"] == "FAKE1"
    assert (tmp_path / "case_nvidia.md").exists()
