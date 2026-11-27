"""Owner: data session. Earnings merge and manual template (task D9, dossier §5.6, §6.4)."""

import pandas as pd

from kpr.collect.earnings import (
    COLUMNS,
    build_template,
    calendar_quarter,
    incomplete_years,
    merge,
)
from kpr.schemas import validate

COMPANIES = [{"company_id": "US1", "region": "US", "ticker": "US1", "sec_ciks": [1]},
             {"company_id": "EU1", "region": "EU", "ticker": "EU1.PA"}]


def test_merge_keeps_edgar_and_dated_manual_rows_only():
    edgar = pd.DataFrame({"company_id": ["US1"], "date": pd.to_datetime(["2020-01-30"]),
                          "quarter": ["Q4"], "source": ["edgar_8k_202"]})
    manual = pd.DataFrame([["EU1", "2020-03-01", "", "", "bob", ""],
                           ["EU1", "", "", "manual", "", "2021: add every release"]],
                          columns=COLUMNS)
    df = merge(edgar, manual, "2014-01-01", "2025-12-31")
    validate(df, "earnings")
    assert df["company_id"].tolist() == ["EU1", "US1"]
    assert df.loc[0, "source"] == "manual" and pd.isna(df.loc[0, "quarter"])


def test_template_keeps_existing_rows_and_marks_empty_years(monkeypatch):
    import kpr.collect.earnings as mod

    monkeypatch.setattr(mod, "yfinance_dates", lambda t, s, e: pd.DataFrame(
        {"date": pd.to_datetime(["2015-03-10"])}))
    existing = pd.DataFrame([["EU1", "2014-07-30", "Q2", "manual", "ann", "H1"]],
                            columns=COLUMNS)
    out = build_template(COMPANIES, existing, "2014-01-01", "2016-12-31")
    assert set(out["company_id"]) == {"EU1"}  # US1 is covered by EDGAR
    dated = out[out["date"] != ""]
    assert dated["date"].tolist() == ["2014-07-30", "2015-03-10"]
    assert dated.iloc[0]["checked_by"] == "ann"  # a checked row is never overwritten
    assert out.loc[out["date"] == "", "note"].str.startswith("2016").all()


def test_incomplete_years_skips_the_first_partial_year():
    df = pd.DataFrame({"company_id": ["EU1"] * 3,
                       "date": pd.to_datetime(["2023-11-01", "2024-03-01", "2024-07-30"])})
    odd = incomplete_years(df, COMPANIES, "2025-12-31")
    assert odd.to_dict("records") == [
        {"company_id": "EU1", "year": 2025, "releases": 0},
        {"company_id": "US1", "year": "all", "releases": 0}]


def test_calendar_quarter_handles_late_annual_results():
    d = pd.Series(pd.to_datetime(["2020-03-15", "2020-01-10", "2020-05-15", "2020-10-12"]))
    assert calendar_quarter(d).tolist() == ["Q4", "Q4", "Q1", "Q3"]
