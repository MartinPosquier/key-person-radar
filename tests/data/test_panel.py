"""Owner: data session. Daily panel (task D11, dossier §6.5, §6.6, §7.1)."""

import numpy as np
import pandas as pd

from kpr.build.panel import (
    attention_z,
    build_panel,
    collapse_to_sessions,
    earnings_flags,
    map_to_sessions,
    select_persons,
)
from kpr.schemas import validate

FIX = "tests/fixtures/"


def test_attention_z_uses_days_t65_to_t6_only():
    days = pd.date_range("2020-01-01", periods=100)
    x = pd.Series(100.0, index=days)
    x.iloc[::2] = 120.0  # some variance in the reference window
    x.iloc[95:] = 10_000.0  # the shock and the 4 days after it
    z = attention_z(x)
    assert z.iloc[95] > 10
    # Day 99 still uses days 34..93 as reference: the shock days 95-98 are in the gap.
    assert z.iloc[99] > 10
    assert np.isnan(z.iloc[20])  # not enough history


def test_weekend_days_map_to_monday_and_the_strongest_day_wins():
    sessions = pd.to_datetime(["2020-01-03", "2020-01-06"])  # Friday, Monday
    days = pd.date_range("2020-01-03", "2020-01-06")
    j0 = map_to_sessions(days, sessions)
    assert j0.dt.strftime("%a").tolist() == ["Fri", "Mon", "Mon", "Mon"]
    daily = pd.DataFrame({"views": [1, 50, 2, 3], "z_wiki": [0.1, 5.0, 0.2, 0.3],
                          "n_articles": [0, 9, 1, 1], "z_news": 0.0, "tone_rel": 0.0},
                         index=days)
    out = collapse_to_sessions(daily, sessions)
    assert out.loc["2020-01-06", "views"] == 50  # Saturday's spike lands on Monday


def test_earnings_flags_cover_plus_minus_k_sessions():
    sessions = pd.bdate_range("2020-01-01", periods=10)
    flags = earnings_flags(sessions, pd.Series(pd.to_datetime(["2020-01-04"])), 1)
    # Saturday release -> J0 Monday 2020-01-06 (position 3): positions 2..4 flagged
    assert np.flatnonzero(flags).tolist() == [2, 3, 4]


def test_selection_applies_views_and_trends_thresholds():
    spans = pd.DataFrame({"person_id": ["a", "b", "c"], "company_id": "X",
                          "start": pd.Timestamp("2020-01-01"), "end": pd.Timestamp("2020-12-31")})
    days = pd.date_range("2020-01-01", "2020-12-31")
    wiki = pd.concat([pd.DataFrame({"page_id": p, "date": days, "views_total": v})
                      for p, v in (("a", 500.0), ("b", 100.0), ("c", 900.0))])
    trends = pd.DataFrame({"person_id": ["c"], "year": [2020], "ratio_person_company": [0.05]})
    sel = select_persons(spans, wiki, trends, 200, 0.10).set_index("person_id")
    assert sel["selected"].to_dict() == {"a": True, "b": False, "c": False}


def test_panel_on_fixtures_matches_the_frozen_schema():
    wiki = pd.read_parquet(FIX + "wiki_views.parquet")
    gdelt = pd.read_parquet(FIX + "gdelt_daily.parquet")
    prices = pd.read_parquet(FIX + "prices.parquet")
    vix = pd.read_parquet(FIX + "vix.parquet")
    earnings = pd.read_parquet(FIX + "earnings.parquet")
    mandates = pd.read_parquet(FIX + "mandates.parquet")
    start, end = str(prices.date.min().date()), str(prices.date.max().date())
    spans = mandates.assign(start=mandates.start.clip(lower=pd.Timestamp(start)),
                            end=mandates.end.fillna(pd.Timestamp(end)))
    spans = spans[spans.start <= spans.end][["person_id", "company_id", "start", "end"]]
    exchanges = {c: "XNYS" for c in prices.company_id.unique()}
    panel = build_panel(spans, wiki, gdelt, prices, vix, earnings, exchanges, start, end, 1)
    validate(panel, "panel")
    assert panel.person_id.nunique() == spans.person_id.nunique()
    assert panel.earnings_window.any() and panel.z_wiki.notna().any()
