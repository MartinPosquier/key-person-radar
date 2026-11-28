"""Owner: data session. Price table construction (task D3, dossier §5.4, §6.4)."""

import numpy as np
import pandas as pd

from kpr.collect.prices import build_index, build_prices, clean_shares, split_adjust_shares
from kpr.schemas import validate


def _hist(closes, splits=None):
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="B", name="date")
    return pd.DataFrame({"Close": closes, "Adj Close": closes, "Volume": 1000.0,
                         "Stock Splits": splits or [0.0] * len(closes)}, index=idx)


def test_split_adjust_multiplies_counts_reported_before_the_split():
    shares = pd.Series([100.0, 100.0, 400.0],
                       index=pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-10"]))
    splits = pd.Series([4.0], index=pd.to_datetime(["2020-01-06"]))
    out = split_adjust_shares(shares, splits)
    assert out.tolist() == [400.0, 400.0, 400.0]


def test_clean_shares_drops_an_isolated_reporting_error():
    shares = pd.Series([100.0] * 6 + [1.0] + [100.0] * 6,
                       index=pd.date_range("2020-01-01", periods=13))
    assert 1.0 not in clean_shares(shares).tolist()


def test_build_prices_log_returns_and_forward_filled_shares():
    hist = _hist([10.0, 11.0, 9.9, 9.9])
    shares = pd.Series([5.0], index=[hist.index[1]])
    df = build_prices("X", hist, shares)
    validate(df, "prices")
    assert np.isnan(df.loc[0, "ret"])
    assert np.isclose(df.loc[1, "ret"], np.log(1.1))
    assert np.isnan(df.loc[0, "shares_out"]) and df.loc[3, "shares_out"] == 5.0


def test_build_index_matches_market_schema():
    validate(build_index("^GSPC", _hist([100.0, 101.0, 99.0])), "market")
