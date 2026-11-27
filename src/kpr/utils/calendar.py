"""Trading calendars per exchange and UTC-day to J0 alignment.

Owner: quant session. Task Q1 in PARALLEL_SESSIONS.md §7. Dossier §6.5.
Library module (no command). Public functions to implement are listed below.
"""

from __future__ import annotations

import pandas as pd


def trading_days(exchange: str, start: str, end: str) -> pd.DatetimeIndex:
    """Trading sessions of `exchange` between `start` and `end` (inclusive)."""
    raise NotImplementedError("utils.calendar.trading_days: quant session, task Q1")


def j0_for(utc_day: pd.Timestamp, exchange: str) -> pd.Timestamp:
    """First trading session of `exchange` on or after the UTC day `utc_day` (dossier §6.5)."""
    raise NotImplementedError("utils.calendar.j0_for: quant session, task Q1")
