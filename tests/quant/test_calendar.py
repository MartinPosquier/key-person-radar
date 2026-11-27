"""Owner: quant session. Unit tests for kpr.utils.calendar (task Q1, dossier §6.5)."""

import numpy as np
import pandas as pd
import pytest

from kpr.utils import calendar as cal

T = pd.Timestamp


# ---- trading days ----------------------------------------------------------------------------

def test_trading_days_skip_weekend_and_holiday():
    # Thu 2024-07-04 is Independence Day: NYSE closed.
    days = cal.trading_days("XNYS", "2024-07-01", "2024-07-08")
    assert list(days) == [T("2024-07-01"), T("2024-07-02"), T("2024-07-03"),
                          T("2024-07-05"), T("2024-07-08")]
    assert days.tz is None


def test_trading_days_inclusive_bounds_and_empty_range():
    assert len(cal.trading_days("XPAR", "2024-01-10", "2024-01-10")) == 1
    assert len(cal.trading_days("XPAR", "2024-01-13", "2024-01-14")) == 0  # weekend
    assert len(cal.trading_days("XPAR", "2024-02-01", "2024-01-01")) == 0


def test_about_252_sessions_a_year():
    for exchange in ("XNYS", "XPAR", "XLON", "XETR"):
        n = len(cal.trading_days(exchange, "2019-01-01", "2019-12-31"))
        assert 248 <= n <= 256, (exchange, n)


def test_nasdaq_shares_the_nyse_calendar():
    a = cal.trading_days("XNAS", "2015-01-01", "2025-12-31")
    b = cal.trading_days("XNYS", "2015-01-01", "2025-12-31")
    assert a.equals(b)


def test_unknown_exchange():
    with pytest.raises(ValueError, match="unknown exchange"):
        cal.trading_days("XTKS", "2024-01-01", "2024-01-31")


# ---- J0 alignment (dossier §6.5, rule 2) ------------------------------------------------------

def test_j0_is_the_day_itself_on_a_trading_day():
    assert cal.j0_for("2024-01-10", "XNYS") == T("2024-01-10")


@pytest.mark.parametrize("day", ["2024-03-09", "2024-03-10"])
def test_j0_weekend_moves_to_monday(day):
    assert cal.j0_for(day, "XNYS") == T("2024-03-11")


def test_j0_holiday_differs_by_exchange():
    # Good Friday 2024-03-29 closes New York, Paris and London; Easter Monday 2024-04-01
    # closes Paris and London only.
    assert cal.j0_for("2024-03-29", "XNYS") == T("2024-04-01")
    assert cal.j0_for("2024-03-29", "XPAR") == T("2024-04-02")
    assert cal.j0_for("2024-03-29", "XLON") == T("2024-04-02")
    # Labour Day closes Euronext, not New York.
    assert cal.j0_for("2024-05-01", "XPAR") == T("2024-05-02")
    assert cal.j0_for("2024-05-01", "XNYS") == T("2024-05-01")
    # Boxing Day closes Paris and London, not New York.
    assert cal.j0_for("2023-12-25", "XNYS") == T("2023-12-26")
    assert cal.j0_for("2023-12-25", "XPAR") == T("2023-12-27")
    assert cal.j0_for("2023-12-25", "XLON") == T("2023-12-27")


def test_j0_holiday_after_weekend():
    # Sat 2024-01-13, Sun 14, Mon 15 = Martin Luther King Day.
    assert cal.j0_for("2024-01-13", "XNYS") == T("2024-01-16")


def test_j0_accepts_intraday_and_tz_aware_values():
    # The UTC day is what counts: 23:30 in New York on Jan 9 is already Jan 10 in UTC.
    ts = T("2024-01-09 23:30", tz="America/New_York")
    assert cal.j0_for(ts, "XNYS") == T("2024-01-10")
    assert cal.j0_for(T("2024-01-10 23:59"), "XNYS") == T("2024-01-10")


def test_j0_series_matches_scalar_version():
    rng = np.random.default_rng(0)
    days = pd.Series(pd.to_datetime("2015-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 11, 300), unit="D"))
    exchanges = pd.Series(rng.choice(["XNYS", "XPAR", "XLON", "XETR", "XSWX"], 300))
    got = cal.j0_series(days, exchanges)
    expected = [cal.j0_for(d, e) for d, e in zip(days, exchanges, strict=True)]
    assert list(got) == expected
    assert got.name == "j0"


def test_j0_series_with_a_single_exchange():
    got = cal.j0_series(pd.Series(["2024-03-09", "2024-03-11"]), "XNYS")
    assert list(got) == [T("2024-03-11"), T("2024-03-11")]


def test_out_of_calendar_raises():
    with pytest.raises(ValueError, match="outside"):
        cal.j0_for("2005-01-01", "XNYS")


# ---- offsets and windows ---------------------------------------------------------------------

def test_session_offset_skips_holidays():
    assert cal.session_offset("2024-07-03", 1, "XNYS") == T("2024-07-05")
    assert cal.session_offset("2024-07-05", -1, "XNYS") == T("2024-07-03")
    assert cal.session_offset("2024-07-05", 0, "XNYS") == T("2024-07-05")


def test_session_offset_rejects_non_sessions():
    with pytest.raises(ValueError, match="not a XNYS session"):
        cal.session_offset("2024-07-04", 1, "XNYS")


def test_reaction_and_estimation_windows():
    w = cal.window("2024-01-10", 0, 2, "XNYS")
    assert list(w) == [T("2024-01-10"), T("2024-01-11"), T("2024-01-12")]
    # Window over a weekend and a holiday: Fri 12, (MLK Mon 15), Tue 16, Wed 17.
    assert list(cal.window("2024-01-12", 0, 2, "XNYS")) == [
        T("2024-01-12"), T("2024-01-16"), T("2024-01-17")]
    pre = cal.window("2024-01-10", -5, -1, "XNYS")
    assert len(pre) == 5 and pre[-1] == T("2024-01-09")
    est = cal.window("2024-06-03", -250, -30, "XNYS")
    assert len(est) == 221
    with pytest.raises(ValueError):
        cal.window("2024-01-10", 2, 0, "XNYS")


# ---- closes and after-close news (dossier §6.5, rules 3 and 5) -------------------------------

def test_session_close_follows_daylight_saving_time():
    assert cal.session_close_utc("2024-01-10", "XNYS") == T("2024-01-10 21:00", tz="UTC")
    assert cal.session_close_utc("2024-07-10", "XNYS") == T("2024-07-10 20:00", tz="UTC")
    assert cal.session_close_utc("2024-01-10", "XPAR") == T("2024-01-10 16:30", tz="UTC")
    assert cal.session_close_utc("2024-07-10", "XPAR") == T("2024-07-10 15:30", tz="UTC")
    # US already on summer time, Europe not yet (2024-03-10 to 2024-03-31).
    assert cal.session_close_utc("2024-03-15", "XNYS") == T("2024-03-15 20:00", tz="UTC")
    assert cal.session_close_utc("2024-03-15", "XLON") == T("2024-03-15 16:30", tz="UTC")


def test_session_close_on_half_day():
    # 2024-07-03: NYSE closes at 13:00 New York time.
    assert cal.session_close_utc("2024-07-03", "XNYS") == T("2024-07-03 17:00", tz="UTC")


def test_news_before_close_moves_same_session():
    assert cal.reaction_session("2024-01-10 20:59", "XNYS") == T("2024-01-10")
    assert cal.reaction_session("2024-01-10 03:00", "XNYS") == T("2024-01-10")


def test_news_after_close_moves_next_session():
    assert cal.reaction_session("2024-01-10 21:00", "XNYS") == T("2024-01-11")
    assert cal.reaction_session("2024-01-10 23:45", "XNYS") == T("2024-01-11")
    # Friday after the close, then a weekend and Martin Luther King Day.
    assert cal.reaction_session("2024-01-12 22:00", "XNYS") == T("2024-01-16")
    # Same timestamp, different exchanges: after the Paris close, before the New York close.
    assert cal.reaction_session("2024-01-10 17:00", "XPAR") == T("2024-01-11")
    assert cal.reaction_session("2024-01-10 17:00", "XNYS") == T("2024-01-10")
    # Half day: 18:00 UTC is after the 13:00 New York close of 2024-07-03.
    assert cal.reaction_session("2024-07-03 18:00", "XNYS") == T("2024-07-05")


def test_after_close_news_is_inside_the_reaction_window():
    # Rule 3: the daily J0 is the UTC day, and J0..J0+2 still covers an after-close story.
    j0 = cal.j0_for("2024-01-10", "XNYS")
    reaction = cal.reaction_session("2024-01-10 22:00", "XNYS")
    assert reaction in cal.window(j0, 0, 2, "XNYS")


def test_reaction_session_accepts_tz_aware_timestamps():
    ts = T("2024-01-10 16:30", tz="America/New_York")  # 21:30 UTC, after the close
    assert cal.reaction_session(ts, "XNYS") == T("2024-01-11")


# ---- ticker suffixes ---------------------------------------------------------------------------

@pytest.mark.parametrize("ticker, exchange", [
    ("TSLA", "XNYS"), ("BRK-B", "XNYS"), ("MC.PA", "XPAR"), ("SAP.DE", "XETR"),
    ("ASML.AS", "XAMS"), ("BP.L", "XLON"), ("STLAM.MI", "XMIL"), ("NESN.SW", "XSWX"),
    ("NOVO-B.CO", "XCSE"), ("RYA.IR", "XDUB"),
])
def test_exchange_for_ticker(ticker, exchange):
    assert cal.exchange_for_ticker(ticker) == exchange


def test_exchange_for_unknown_suffix():
    with pytest.raises(ValueError):
        cal.exchange_for_ticker("7203.T")
