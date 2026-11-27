"""Download adjusted prices, market indices and VIX (yfinance), cached.

Owner: data session. Task D3 in PARALLEL_SESSIONS.md §7. Dossier §5.4, §6.4.

- Raw cache: raw/yfinance/<ticker>_history.parquet (daily bars with actions, as returned by
  yfinance with auto_adjust=False) and <ticker>_shares.parquet (shares outstanding as
  reported). A ticker already cached is never downloaded again (use --refresh to force).
- Tables written: `prices` (one row per company and trading day), `market` (^GSPC and
  ^STOXX), `vix`.
- close_adj is the close adjusted for dividends and splits; ret is the log return of
  close_adj between consecutive trading days (dossier §6.4).
- shares_out is split-adjusted, so that close (split-adjusted by yfinance) × shares_out is
  the market value. yfinance reports shares outstanding only from about 2015; before the
  first report shares_out is missing.

Usage: python -m kpr.collect.prices [--refresh] [--only TICKER ...]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import yfinance as yf

from kpr.collect.common import append_collection_log, load_params, load_universe
from kpr.utils.io import raw_dir, write_table
from kpr.utils.logging import get_logger

log = get_logger(__name__)

SOURCE = "yfinance"
VIX_TICKER = "^VIX"
SHARES_OUTLIER = 0.5  # drop a share count more than 50 % away from its rolling median


def _safe(ticker: str) -> str:
    return ticker.replace("^", "IDX_").replace(".", "_")


def _write_parquet(df: pd.DataFrame, path) -> None:
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def _period() -> tuple[str, str]:
    p = load_params()["period"]
    end_excl = (pd.Timestamp(p["end"]) + pd.DateOffset(days=1)).date().isoformat()
    return p["start"], end_excl


def fetch_history(ticker: str, refresh: bool = False) -> pd.DataFrame:
    """Daily bars with Close, Adj Close, Volume, Stock Splits; index = local trading date."""
    path = raw_dir(SOURCE) / f"{_safe(ticker)}_history.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    start, end = _period()
    hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False, actions=True)
    if hist.empty:
        raise RuntimeError(f"yfinance returned no prices for {ticker}")
    hist.index = pd.DatetimeIndex(hist.index.tz_localize(None).normalize(), name="date")
    _write_parquet(hist, path)
    append_collection_log(SOURCE, kind="history", ticker=ticker, rows=len(hist),
                          first=hist.index.min(), last=hist.index.max())
    time.sleep(0.5)
    return hist


def fetch_shares(ticker: str, refresh: bool = False) -> pd.Series:
    """Shares outstanding as reported (not split-adjusted); index = date."""
    path = raw_dir(SOURCE) / f"{_safe(ticker)}_shares.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)["shares"]
    start, end = _period()
    try:
        s = yf.Ticker(ticker).get_shares_full(start=start, end=end)
    except Exception as exc:  # noqa: BLE001 - yfinance raises many kinds of errors here
        log.warning("no share count for %s: %s", ticker, exc)
        s = None
    if s is None or len(s) == 0:
        s = pd.Series(dtype=float)
    else:
        s.index = pd.DatetimeIndex(s.index.tz_localize(None).normalize())
    s = s.groupby(level=0).last().rename("shares")
    s.index.name = "date"
    _write_parquet(s.to_frame(), path)
    append_collection_log(SOURCE, kind="shares", ticker=ticker, rows=len(s))
    time.sleep(0.5)
    return s


def split_adjust_shares(shares: pd.Series, splits: pd.Series) -> pd.Series:
    """Express each reported share count in today's share units.

    A count reported before a 4:1 split is multiplied by 4, so that it matches the
    split-adjusted close that yfinance returns.
    """
    splits = splits[splits > 0].sort_index()
    if shares.empty or splits.empty:
        return shares.astype(float)
    factor = pd.Series(1.0, index=shares.index)
    for when, ratio in splits.items():
        factor[shares.index < when] *= ratio
    return shares.astype(float) * factor


def clean_shares(shares: pd.Series) -> pd.Series:
    """Drop isolated reporting errors: points far from the rolling median of their neighbours."""
    if len(shares) < 5:
        return shares
    med = shares.rolling(15, center=True, min_periods=3).median()
    keep = (shares / med - 1).abs() <= SHARES_OUTLIER
    return shares[keep]


def build_prices(company_id: str, hist: pd.DataFrame, shares: pd.Series) -> pd.DataFrame:
    hist = hist[hist["Close"].notna() & (hist["Close"] > 0)]
    adj = hist["Adj Close"].astype(float)
    splits = hist.get("Stock Splits", pd.Series(dtype=float))
    sh = clean_shares(split_adjust_shares(shares.sort_index(), splits))
    shares_daily = sh.reindex(hist.index.union(sh.index)).ffill().reindex(hist.index)
    return pd.DataFrame({
        "company_id": company_id,
        "date": hist.index,
        "close_adj": adj.values,
        "volume": hist["Volume"].astype(float).values,
        "shares_out": shares_daily.values,
        "ret": np.log(adj).diff().values,
    })


def build_index(index_id: str, hist: pd.DataFrame) -> pd.DataFrame:
    hist = hist[hist["Close"].notna() & (hist["Close"] > 0)]
    close = hist["Close"].astype(float)
    return pd.DataFrame({"index_id": index_id, "date": hist.index, "close": close.values,
                         "ret": np.log(close).diff().values})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kpr.collect.prices")
    parser.add_argument("--refresh", action="store_true", help="download again")
    parser.add_argument("--only", nargs="*", help="limit to these tickers (no table written)")
    args = parser.parse_args(argv)

    companies = load_universe()
    if args.only:
        for t in args.only:
            fetch_history(t, args.refresh)
        return 0

    frames, errors = [], []
    for c in companies:
        try:
            hist = fetch_history(c["ticker"], args.refresh)
            shares = fetch_shares(c["ticker"], args.refresh)
            for old in c.get("former_tickers", []):
                older = fetch_shares(old, args.refresh)
                if len(shares):
                    older = older[older.index < shares.index.min()]
                shares = pd.concat([older, shares])
            frames.append(build_prices(c["company_id"], hist, shares))
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            errors.append(f"{c['ticker']}: {exc}")
            log.error("failed %s: %s", c["ticker"], exc)
    prices = pd.concat(frames, ignore_index=True)
    write_table(prices, "prices")

    indices = sorted({c["market_index"] for c in companies})
    market = pd.concat([build_index(i, fetch_history(i, args.refresh)) for i in indices],
                       ignore_index=True)
    write_table(market, "market")

    vix_hist = fetch_history(VIX_TICKER, args.refresh)
    vix = pd.DataFrame({"date": vix_hist.index, "vix": vix_hist["Close"].astype(float).values})
    write_table(vix[vix["vix"].notna()].reset_index(drop=True), "vix")

    append_collection_log(SOURCE, kind="tables", prices=len(prices), market=len(market),
                          vix=len(vix), errors=errors)
    missing_shares = prices.groupby("company_id")["shares_out"].apply(lambda s: s.isna().mean())
    log.info("prices: %d rows, %d companies; market: %d rows; vix: %d rows",
             len(prices), prices["company_id"].nunique(), len(market), len(vix))
    log.info("share of days without shares_out: median %.0f %%, max %.0f %% (%s)",
             100 * missing_shares.median(), 100 * missing_shares.max(), missing_shares.idxmax())
    if errors:
        log.error("%d tickers failed: %s", len(errors), errors)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
