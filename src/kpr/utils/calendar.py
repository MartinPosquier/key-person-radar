"""Trading calendars per exchange and UTC-day to J0 alignment.

Owner: quant session. Task Q1 in PARALLEL_SESSIONS.md §7. Dossier §6.5.
Library module (no command).

Alignment rules (dossier §6.5):

1. The shock day `d` is the UTC day of the attention peak (Wikipedia and GDELT days run
   from midnight to midnight UTC).
2. J0 is the first session of the company's exchange that starts or is still running after
   the start of day `d`: J0 = d if d is a trading day, otherwise the next session.
3. The reaction window J0..J0+2 covers news that breaks after the close of J0.
4. The anticipation check uses J-5..J-1.
5. For timestamped news (GDELT, 15-minute precision), `reaction_session` gives the first
   session whose close comes after the timestamp, to check on a sample whether articles land
   before or after the close.

Calendars come from the `exchange_calendars` package (holidays, half days, daylight saving
time). Exchange codes are the ISO MIC codes used in `config/universe.yaml`.

All session dates returned here are tz-naive midnight timestamps, the same convention as the
`date` columns of the frozen tables.
"""

from __future__ import annotations

from functools import cache

import exchange_calendars as xcals
import pandas as pd

# Codes allowed in config/universe.yaml.
SUPPORTED_EXCHANGES = (
    "XNYS", "XNAS", "XPAR", "XETR", "XAMS", "XLON", "XMIL", "XSWX", "XCSE", "XDUB",
)
# Nasdaq has the NYSE holidays and hours. Map it in case the installed exchange_calendars
# version does not ship an XNAS calendar.
_CALENDAR_FALLBACK = {"XNAS": "XNYS"}

# First session we may ever need: 2014 events with an estimation window of 250 sessions
# start in 2013 (dossier §7.5); the margin keeps the 2014-2025 study safe.
CALENDAR_START = "2010-01-01"

# yfinance ticker suffix -> exchange, used only as a fallback when universe.yaml has no
# `exchange` field. US tickers (no suffix) map to XNYS: same calendar as Nasdaq.
_SUFFIX_TO_EXCHANGE = {
    "": "XNYS",
    "PA": "XPAR",
    "DE": "XETR",
    "AS": "XAMS",
    "L": "XLON",
    "MI": "XMIL",
    "SW": "XSWX",
    "CO": "XCSE",
    "IR": "XDUB",
}


@cache
def get_calendar(exchange: str) -> xcals.ExchangeCalendar:
    """Cached `exchange_calendars` calendar for `exchange` (ISO MIC code)."""
    code = exchange.strip().upper()
    if code not in SUPPORTED_EXCHANGES:
        raise ValueError(f"unknown exchange {exchange!r}; supported: {SUPPORTED_EXCHANGES}")
    if code not in xcals.get_calendar_names(include_aliases=True):
        code = _CALENDAR_FALLBACK[code]
    return xcals.get_calendar(code, start=CALENDAR_START)


def _day(value) -> pd.Timestamp:
    """Normalise a date-like value to a tz-naive midnight timestamp (the UTC day)."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _naive(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _in_bounds(cal: xcals.ExchangeCalendar, day: pd.Timestamp) -> None:
    first, last = _naive(cal.first_session), _naive(cal.last_session)
    if day < first or day > last:
        raise ValueError(
            f"{day.date()} is outside the {cal.name} calendar ({first.date()} to {last.date()})"
        )


def _session(cal: xcals.ExchangeCalendar, value, exchange: str) -> pd.Timestamp:
    s = _day(value)
    _in_bounds(cal, s)
    if not cal.is_session(s):
        raise ValueError(f"{s.date()} is not a {exchange} session")
    return s


def exchange_for_ticker(ticker: str) -> str:
    """Exchange of a yfinance ticker from its suffix (fallback when universe.yaml has none)."""
    suffix = ticker.rsplit(".", 1)[1].upper() if "." in ticker else ""
    if suffix not in _SUFFIX_TO_EXCHANGE:
        raise ValueError(f"cannot infer the exchange of ticker {ticker!r}")
    return _SUFFIX_TO_EXCHANGE[suffix]


@cache
def _all_sessions(exchange: str) -> pd.DatetimeIndex:
    cal = get_calendar(exchange)
    sessions = cal.sessions
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    return pd.DatetimeIndex(sessions, name="date")


def trading_days(exchange: str, start, end) -> pd.DatetimeIndex:
    """Trading sessions of `exchange` between `start` and `end` (inclusive)."""
    sessions = _all_sessions(exchange)
    lo = sessions.searchsorted(_day(start), side="left")
    hi = sessions.searchsorted(_day(end), side="right")
    return sessions[lo:hi]


def is_trading_day(day, exchange: str) -> bool:
    """True if the UTC day `day` is a trading session of `exchange`."""
    cal = get_calendar(exchange)
    d = _day(day)
    _in_bounds(cal, d)
    return bool(cal.is_session(d))


def j0_for(utc_day, exchange: str) -> pd.Timestamp:
    """First trading session of `exchange` on or after the UTC day `utc_day` (dossier §6.5)."""
    cal = get_calendar(exchange)
    d = _day(utc_day)
    _in_bounds(cal, d)
    return _naive(cal.date_to_session(d, direction="next"))


def j0_series(utc_days: pd.Series, exchanges: pd.Series | str) -> pd.Series:
    """Vectorised `j0_for`: one J0 per row, with one exchange per row or one for all rows."""
    days = pd.to_datetime(pd.Series(utc_days)).dt.normalize()
    if isinstance(exchanges, str):
        exchanges = pd.Series(exchanges, index=days.index)
    out = pd.Series(pd.NaT, index=days.index, dtype="datetime64[ns]", name="j0")
    for exchange, idx in exchanges.groupby(exchanges).groups.items():
        sessions = _all_sessions(exchange)
        values = days.loc[idx].to_numpy()
        if (values < sessions[0].to_datetime64()).any():
            raise ValueError(f"some days are before the first {exchange} session")
        pos = sessions.searchsorted(values, side="left")
        if (pos >= len(sessions)).any():
            raise ValueError(f"some days are after the last {exchange} session")
        out.loc[idx] = sessions[pos].to_numpy()
    return out


def session_offset(session, k: int, exchange: str) -> pd.Timestamp:
    """The session `k` trading sessions after `session` (before it if k < 0): J0+k."""
    cal = get_calendar(exchange)
    s = _session(cal, session, exchange)
    return _naive(cal.session_offset(s, k))


def window(j0, lo: int, hi: int, exchange: str) -> pd.DatetimeIndex:
    """Sessions J0+lo .. J0+hi, inclusive (for example [0, 2] or [-250, -30])."""
    if lo > hi:
        raise ValueError(f"empty window [{lo}, {hi}]")
    first = session_offset(j0, lo, exchange)
    last = session_offset(j0, hi, exchange)
    return trading_days(exchange, first, last)


def session_close_utc(session, exchange: str) -> pd.Timestamp:
    """Closing time of `session` in UTC (handles daylight saving time and half days)."""
    cal = get_calendar(exchange)
    s = _session(cal, session, exchange)
    close = pd.Timestamp(cal.session_close(s))
    return close.tz_localize("UTC") if close.tzinfo is None else close.tz_convert("UTC")


def reaction_session(timestamp_utc, exchange: str) -> pd.Timestamp:
    """First session whose close comes strictly after a precise UTC timestamp.

    News published before the close of a trading day can move that day's close; news
    published at or after the close (or on a non-trading day) can only move the next session.
    A naive timestamp is read as UTC.
    """
    ts = pd.Timestamp(timestamp_utc)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    day = ts.tz_localize(None).normalize()
    session = j0_for(day, exchange)
    if session == day and ts >= session_close_utc(session, exchange):
        session = session_offset(session, 1, exchange)
    return session
