"""Owner: ml session. Tests for kpr.scores.export_app (task M13, dossier §12)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from kpr.scores import export_app as X

APP_DIR = Path(__file__).resolve().parents[2] / "app"
sys.path.insert(0, str(APP_DIR))

import common  # noqa: E402

PERSONS = [{"person_id": "p_alpha", "wiki_pages": {"en": "Alex_Alpha"}}]
COMPANIES = [{"company_id": "FAKE1", "name": "Fakecorp", "region": "US", "sector": "Semis",
              "market_index": "^GSPC", "group": "media"},
             {"company_id": "FAKE3", "name": "Synth", "region": "EU", "sector": "Energy",
              "market_index": "^STOXX", "group": "control"}]


@pytest.fixture(scope="module")
def tables(monkeypatch_module):
    monkeypatch_module.setenv("KPR_USE_FIXTURES", "1")
    from kpr.utils.io import read_table

    return {n: read_table(n) for n in ("scores", "events", "mandates", "panel", "prices",
                                       "market")}


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def exported(tables, tmp_path_factory):
    out = tmp_path_factory.mktemp("app_data")
    car_pred = pd.Series(0.01, index=tables["events"]["event_id"])
    meta = X.export(out, tables, PERSONS, COMPANIES, car_pred, None, "test", demo=False)
    return out, meta


def test_every_app_file_matches_what_the_pages_read(exported):
    out, _ = exported
    for name in common.APP_FILES:
        common.load_frame(name, data_dir=out)
    assert not (out / "case_study.json").exists()


def test_entities_use_config_names_and_fall_back_to_ids(exported):
    out, _ = exported
    ent = pd.read_parquet(out / "entities.parquet").set_index("person_id")
    assert ent.loc["p_alpha", "person_name"] == "Alex Alpha"
    assert ent.loc["p_beta", "person_name"] == "P Beta"
    assert ent.loc["p_alpha", "company_name"] == "Fakecorp"
    assert ent.loc["p_gamma", "region"] == "unknown"  # FAKE2 not in the test universe


def test_daily_abnormal_return_is_market_adjusted(exported, tables):
    out, _ = exported
    daily = pd.read_parquet(out / "daily.parquet")
    row = daily.query("person_id == 'p_alpha'").iloc[10]
    px = tables["prices"].query("company_id == 'FAKE1' and date == @row.date")["ret"].iloc[0]
    mk = tables["market"].query("index_id == '^GSPC' and date == @row.date")["ret"].iloc[0]
    assert row["ar"] == pytest.approx(px - mk)


def test_meta_records_the_sanity_checks(exported):
    out, meta = exported
    saved = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert saved["demo"] is False
    assert set(saved["checks"]) == {"top10_control_group", "mean_v_by_group",
                                    "low_confidence_person_years", "high_v_low_pi"}
    assert saved["checks"]["top10_control_group"] >= 1  # FAKE3 is a control company here


def test_case_from_quant():
    case = {"person": "jensen_huang", "company": "NVDA", "date": "2026-03-02",
            "date_source": "highest attention z-score", "j0": "2026-03-02",
            "cars": {"NVDA": {"car_0_2": -0.031}}, "news": {"earnings_dates": []},
            "model": {"year": 2025, "sign": "negative", "expected_abs_car": 0.024},
            "verdict": "consistent (actual is 1.3x the expected magnitude)"}
    app_case = X.case_from_quant(case)
    assert app_case["predicted_car"] == pytest.approx(-0.024)
    assert app_case["actual_car"] == pytest.approx(-0.031)
    assert "consistent" in app_case["story"] and app_case["demo"] is False


def test_fixture_export_needs_an_explicit_folder(monkeypatch):
    monkeypatch.setenv("KPR_USE_FIXTURES", "1")
    assert X.main([]) == 2


def test_app_renders_on_an_export(exported, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    out, _ = exported
    monkeypatch.setattr(common, "DATA_DIR", out)
    st.cache_data.clear()
    at = AppTest.from_file(str(APP_DIR / "Home.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not any("Demo mode" in i.value for i in at.info)  # real export: no demo banner
    st.cache_data.clear()
