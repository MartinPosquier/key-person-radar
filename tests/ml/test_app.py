"""Owner: ml session. Tests for the Streamlit app (task M1).

- the helpers of app/common.py on small hand-made frames;
- every page renders without exception on the demo files of app/data/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

APP_DIR = Path(__file__).resolve().parents[2] / "app"
sys.path.insert(0, str(APP_DIR))

import common  # noqa: E402

PAGES = ["Home.py", *sorted(p.relative_to(APP_DIR).as_posix()
                            for p in (APP_DIR / "pages").glob("*.py"))]


def test_every_page_of_the_dossier_exists():
    names = " ".join(PAGES)
    for page in ["Leaderboard", "Company", "Person", "Compare", "Shock_explorer",
                 "Case_study", "Methodology"]:
        assert page in names


def test_demo_files_have_required_columns():
    for name in common.APP_FILES:
        common.load_frame(name)  # raises AppDataError on a missing column


def test_load_frame_reports_missing_file(tmp_path):
    with pytest.raises(common.AppDataError, match="missing"):
        common.load_frame("scores", data_dir=tmp_path)


def test_load_frame_reports_missing_column(tmp_path):
    pd.DataFrame({"person_id": ["a"]}).to_parquet(tmp_path / "entities.parquet")
    with pytest.raises(common.AppDataError, match="lacks columns"):
        common.load_frame("entities", data_dir=tmp_path)


def test_company_dependence_is_max_and_upper_bound_is_sum():
    scores = pd.DataFrame({
        "company_id": ["C", "C", "D"], "year": [2020, 2020, 2020],
        "person_id": ["a", "b", "c"], "v_pct": [3.0, 1.5, 2.0],
    })
    dep = common.company_dependence(scores).set_index("company_id")
    assert dep.loc["C", "dependence_index"] == 3.0
    assert dep.loc["C", "upper_bound"] == 4.5
    assert dep.loc["C", "n_persons"] == 2


def test_latest_year_keeps_most_recent_row():
    scores = pd.DataFrame({"person_id": ["a", "a", "b"], "company_id": ["C", "C", "D"],
                           "year": [2020, 2021, 2020], "v_pct": [1.0, 2.0, 3.0]})
    latest = common.latest_year(scores).set_index("person_id")
    assert latest.loc["a", "year"] == 2021


def test_honest_display_formatting():
    assert common.fmt_v(6.0, 3.0, 9.0) == "6.0% [3.0–9.0%] (estimate)"
    assert "estimate" in common.fmt_v(6.0, float("nan"), float("nan"))
    assert common.fmt_score(10.0) == "10.0 (capped)"
    assert common.fmt_score(4.21) == "4.2"
    assert common.confidence_badge(7) == "low confidence"
    assert common.confidence_badge(8) == ""
    assert common.fmt_usd(2.5e9) == "$2.5bn"


def test_mean_car_curve_cumulates_abnormal_returns():
    dates = pd.bdate_range("2021-01-01", periods=30)
    daily = pd.DataFrame({"company_id": "C", "date": dates, "ar": 0.0})
    daily.loc[15, "ar"] = -0.05  # the shock day
    events = pd.DataFrame({"event_id": ["e"], "company_id": ["C"], "j0": [dates[15]],
                           "sign": ["negative"]})
    curve = common.mean_car_curve(events, daily, window=10).set_index("k")
    assert curve.loc[-1, "car"] == 0.0
    assert curve.loc[0, "car"] == pytest.approx(-0.05)
    assert curve.loc[10, "car"] == pytest.approx(-0.05)
    assert len(curve) == 21


def test_mean_car_curve_skips_events_too_close_to_the_edges():
    dates = pd.bdate_range("2021-01-01", periods=12)
    daily = pd.DataFrame({"company_id": "C", "date": dates, "ar": 0.01})
    events = pd.DataFrame({"event_id": ["e"], "company_id": ["C"], "j0": [dates[2]],
                           "sign": ["positive"]})
    assert common.mean_car_curve(events, daily).empty


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_on_demo_data(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_DIR / page), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(common.DISCLAIMER in w.value for w in at.warning), "disclaimer missing"
