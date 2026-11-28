"""Owner: data session. EDGAR earnings dates (task D6, dossier §5.6, §6.4)."""

import pandas as pd

from kpr.collect.edgar import calendar_quarter, earnings_from_filings
from kpr.schemas import validate


def _f(rows):
    return pd.DataFrame(rows, columns=["accessionNumber", "filingDate", "reportDate", "form",
                                       "items"])


def test_8k_202_uses_report_date_and_ignores_other_items():
    filings = _f([
        ("a", "2019-02-01", "2019-01-31", "8-K", "2.02,9.01"),
        ("b", "2019-03-01", "", "8-K", "5.02"),
        ("c", "2019-05-02", "", "8-K", "2.02"),
        ("d", "2019-05-03", "2019-05-02", "8-K/A", "2.02"),
        ("e", "2019-08-01", "2019-07-31", "8-K", "2.02"),
        ("f", "2019-11-01", "2019-10-31", "8-K", "2.02"),
        ("g", "2019-11-05", "", "10-Q", ""),
    ])
    df = earnings_from_filings("X", filings, "2014-01-01", "2025-12-31")
    validate(df, "earnings")
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2019-01-31", "2019-05-02", "2019-07-31", "2019-10-31"]
    assert set(df["source"]) == {"edgar_8k_202"}  # 4 releases: no periodic fallback


def test_thin_years_fall_back_on_periodic_reports_not_near_an_8k():
    filings = _f([
        ("a", "2019-02-23", "2019-02-23", "8-K", "2.02"),
        ("b", "2019-02-25", "", "10-K", ""),   # 2 days after the 8-K: same release
        ("c", "2019-05-04", "", "10-Q", ""),
        ("d", "2019-08-03", "", "10-Q", ""),
    ])
    df = earnings_from_filings("BRK", filings, "2014-01-01", "2025-12-31")
    assert df["source"].tolist() == ["edgar_8k_202", "edgar_10q_10k", "edgar_10q_10k"]


def test_calendar_quarter_of_reported_period():
    dates = pd.Series(pd.to_datetime(["2020-02-05", "2020-05-06", "2020-07-30", "2020-10-29"]))
    assert calendar_quarter(dates).tolist() == ["Q4", "Q1", "Q2", "Q3"]
