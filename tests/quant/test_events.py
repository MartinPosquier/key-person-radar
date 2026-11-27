"""Owner: quant session. Unit tests for kpr.build.events (task Q4, dossier §7.1-7.4)."""

import numpy as np
import pandas as pd
import pytest

from kpr.build import events as E

FIXTURES = E.REPO_ROOT / "tests" / "fixtures"
T = pd.Timestamp


@pytest.fixture(scope="module")
def tables():
    names = ("panel", "prices", "market", "earnings", "wiki_views", "gdelt_daily", "events")
    return {n: pd.read_parquet(FIXTURES / f"{n}.parquet") for n in names}


def _build(t, **kw):
    return E.build_events(t["panel"], t["prices"], t["market"], t["earnings"], t["wiki_views"],
                          t["gdelt_daily"], universe=kw.pop("universe", {}), **kw)


@pytest.fixture(scope="module")
def built(tables):
    return _build(tables)


# ---- z-score (§7.1) --------------------------------------------------------------------------

def test_attention_z_uses_t_minus_65_to_t_minus_6():
    rng = np.random.default_rng(0)
    v = pd.Series(rng.lognormal(5, 0.2, 100))
    v.iloc[95:] = 1e6  # days t-5 .. t-1 of the last day: must not enter its reference
    z = E.attention_z(v, window=60, gap=5)
    x = np.log1p(v)
    ref = x.iloc[99 - 65: 99 - 5]  # t-65 .. t-6 for t = 99
    assert len(ref) == 60
    assert z.iloc[99] == pytest.approx((x.iloc[99] - ref.mean()) / ref.std())


def test_attention_z_needs_enough_history_and_spread():
    z = E.attention_z(pd.Series(np.arange(30.0)), window=60, gap=5, min_periods=20)
    # First z on day 25: its reference is days 0..19 (20 values, shifted by 6).
    assert z.iloc[:25].isna().all() and z.iloc[25:].notna().all()
    flat = E.attention_z(pd.Series(np.full(80, 100.0)))
    assert flat.isna().all()


# ---- detection and clustering (§7.2) ------------------------------------------------------------

def test_clusters_chain_and_date_on_first_day():
    dates = pd.Series(pd.to_datetime(["2021-01-01", "2021-01-05", "2021-01-12", "2021-01-30",
                                      "2021-02-10"]))
    # Jan 12 is 11 days after Jan 1 but 7 days after Jan 5: same cluster.
    assert list(E.first_of_clusters(dates, 10)) == [True, False, False, True, True]


def test_detect_shocks_applies_the_four_conditions():
    days = pd.date_range("2021-01-01", periods=6)
    panel = pd.DataFrame({
        "person_id": "p", "company_id": "C", "date": days,
        "z_wiki":     [3.5, 2.9, 5.0, 4.0, 3.1, 3.2],
        "z_news":     [2.5, 9.0, 1.9, 2.1, 2.2, 2.3],
        "n_articles": [10, 10, 10, 2, 10, 10],
        "tone_rel":   [0.0] * 6,
    })
    out = E.detect_shocks(panel)
    # Day 0 qualifies; day 1 fails z_wiki, day 2 fails z_news, day 3 fails the article count;
    # days 4 and 5 qualify but are within 10 days of day 0: merged into it.
    assert list(out["date_utc"]) == [days[0]]
    assert out["n_peak_days"].iloc[0] == 3


def test_thresholds_come_from_params():
    days = pd.date_range("2021-01-01", periods=2, freq="30D")
    panel = pd.DataFrame({"person_id": "p", "company_id": "C", "date": days,
                          "z_wiki": [2.6, 3.5], "z_news": [3.0, 3.0], "n_articles": [5, 5],
                          "tone_rel": [0.0, 0.0]})
    assert len(E.detect_shocks(panel)) == 1
    loose = E.load_params({"shocks": {"z_wiki_min": 2.5}})
    assert len(E.detect_shocks(panel, loose)) == 2


def test_sign_thresholds():
    s = E.sign_of(pd.Series([-1.5, -1.0, 0.0, 1.0, 1.01, np.nan]))
    assert list(s) == ["negative", "neutral", "neutral", "neutral", "positive", "neutral"]


# ---- J0 with a real calendar ------------------------------------------------------------------

def test_j0_follows_the_exchange_calendar(tables):
    universe = {"FAKE1": E.Company("FAKE1", "XNYS", "^GSPC")}
    m = E.Markets(tables["prices"], tables["market"], universe)
    assert m.j0("FAKE1", T("2020-07-03")) == T("2020-07-06")  # Independence Day observed
    assert m.j0("FAKE1", T("2020-07-04")) == T("2020-07-06")  # Saturday
    assert m.index_of("FAKE1") == "^GSPC"
    # Without a calendar, the price dates are the sessions (fixtures have no holidays).
    m2 = E.Markets(tables["prices"], tables["market"], {})
    assert m2.j0("FAKE1", T("2020-07-03")) == T("2020-07-03")
    assert m2.index_of("FAKE3") == "^STOXX"


# ---- whole build on the fixtures -------------------------------------------------------------

def test_detected_events_are_the_planted_ones(tables, built):
    planted = set(tables["events"]["event_id"])
    assert set(built["event_id"]) <= planted  # no false positive
    assert len(built) >= 25
    both = built.merge(tables["events"], on="event_id", suffixes=("", "_true"))
    assert (both["sign"] == both["sign_true"]).all()
    kept = both[~both["excluded"]]
    assert kept["car_0_2"].corr(kept["car_0_2_true"]) > 0.5


def test_build_returns_schema_columns(built):
    for col in ("event_id", "person_id", "company_id", "date_utc", "j0", "z_wiki", "z_news",
                "n_articles", "tone_rel", "tone_neg_share", "sign", "car_0_2", "car_m5_m1",
                "car_0_1", "car_0_5", "excluded", "exclusion_reason"):
        assert col in built.columns
    assert built["event_id"].is_unique
    assert (built["j0"] >= built["date_utc"]).all()
    assert built["tone_neg_share"].notna().all()


def test_short_history_is_flagged_not_dropped(built):
    short = built[built["n_est"] < 120]
    assert len(short) > 0
    assert short["excluded"].all()
    assert short["exclusion_reason"].str.contains("short_history").all()
    assert built.loc[~built["excluded"], "exclusion_reason"].isna().all()


def test_earnings_filter(tables, built):
    flagged = built[built["exclusion_reason"].fillna("").str.contains("earnings")]
    assert len(flagged) >= 1
    for ev in flagged.itertuples():
        releases = tables["earnings"].loc[tables["earnings"].company_id == ev.company_id, "date"]
        sessions = pd.DatetimeIndex(sorted(tables["prices"].loc[
            tables["prices"].company_id == ev.company_id, "date"]))
        pos = sessions.get_loc(ev.j0)
        assert min(abs(pos - sessions.searchsorted(r)) for r in releases) <= 1
    wider = _build(tables, params=E.load_params({"filters": {"earnings_window_sessions": 5}}))
    n_wide = wider["exclusion_reason"].fillna("").str.contains("earnings").sum()
    assert n_wide >= len(flagged)


def test_extreme_market_filter(tables, built):
    ev = built[~built["excluded"]].iloc[0]
    index_id = {"FAKE1": "^GSPC", "FAKE2": "^GSPC", "FAKE3": "^STOXX"}[ev.company_id]
    market = tables["market"].copy()
    day1 = tables["prices"].query("company_id == @ev.company_id").date.sort_values()
    day1 = day1[day1 > ev.j0].iloc[0]  # J0 + 1
    market.loc[(market.index_id == index_id) & (market.date == day1), "ret"] = -0.06
    t = dict(tables, market=market)
    out = _build(t).set_index("event_id")
    assert "extreme_market" in out.loc[ev.event_id, "exclusion_reason"]


def test_company_dominance_filter(tables, built):
    ev = built[~built["excluded"]].iloc[0]
    wiki = tables["wiki_views"].copy()
    page = (wiki.page_id == E.company_page_id(ev.company_id)) & (wiki.date == ev.date_utc)
    wiki.loc[page, "views_total"] *= 1e6
    out = _build(dict(tables, wiki_views=wiki)).set_index("event_id")
    assert out.loc[ev.event_id, "z_company"] > out.loc[ev.event_id, "z_wiki"] + 2
    assert "company_dominant" in out.loc[ev.event_id, "exclusion_reason"]


def test_reasons_are_combined(tables, built):
    ev = built[built["exclusion_reason"] == "short_history"].iloc[0]
    wiki = tables["wiki_views"].copy()
    page = (wiki.page_id == E.company_page_id(ev.company_id)) & (wiki.date == ev.date_utc)
    wiki.loc[page, "views_total"] *= 1e6
    out = _build(dict(tables, wiki_views=wiki)).set_index("event_id")
    assert out.loc[ev.event_id, "exclusion_reason"] == "short_history;company_dominant"


def test_no_shock_gives_empty_frame(tables):
    panel = tables["panel"].assign(z_wiki=0.0)
    out = _build(dict(tables, panel=panel))
    assert out.empty and "car_0_2" in out.columns
