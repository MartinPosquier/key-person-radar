"""Abnormal returns: market model, market-adjusted, Fama-French 3; CAR windows.

Owner: quant session. Task Q2 in PARALLEL_SESSIONS.md §7. Dossier §7.5.

Event time. Day k is the k-th session after J0 (k = 0 is J0) in a list of sessions given by
the caller: the exchange calendar of the company (`kpr.utils.calendar.trading_days`) on real
data, or simply the dates of the price table. A session with no price gives a missing return,
never a shifted window.

Models (dossier §7.5):

- ``market``: R_t = alpha + beta R_m,t + e_t, estimated by OLS on [J-250, J-30];
  AR_t = R_t - (alpha + beta R_m,t). This is the main model.
- ``market_adjusted``: beta = 1, no estimation; AR_t = R_t - R_m,t.
- ``ff3``: R_t - rf_t = a + b (Mkt-RF)_t + s SMB_t + h HML_t + e_t on the same window;
  AR_t = (R_t - rf_t) - (a + b (Mkt-RF)_t + s SMB_t + h HML_t).

CAR[lo, hi] = sum of AR_t for k in lo..hi. It is missing if any AR of the window is missing.

Library use (events.py, validation):  ``event_abnormal_returns`` for one event,
``compute_cars`` for a table of events. Command use (``python -m kpr.build.returns``): a
diagnostic that recomputes the CARs of the events table from prices and prints a summary;
it writes no shared table.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kpr.utils.logging import get_logger

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS = ("market", "market_adjusted", "ff3")
FF3_COLUMNS = ("mkt_rf", "smb", "hml", "rf")
# Below this many estimation days an OLS fit is not computed at all. The stricter rule of
# dossier §7.4 (at least 120 sessions, `filters.min_estimation_obs`) is an exclusion applied
# by events.py from the `n_est` column, so excluded shocks still get a CAR for bias checks.
MIN_FIT_OBS = 30

# Output column name for each window of the events table (dossier §7.7).
DEFAULT_WINDOWS: dict[str, tuple[int, int]] = {
    "car_0_2": (0, 2),
    "car_m5_m1": (-5, -1),
    "car_0_1": (0, 1),
    "car_0_5": (0, 5),
}


def load_params() -> dict:
    """The `returns` block of config/params/quant.yaml."""
    with open(REPO_ROOT / "config" / "params" / "quant.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["returns"]


# ---- model fits --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Fit:
    """Estimated normal-return model of one stock around one event."""

    model: str
    params: dict[str, float] = field(default_factory=dict)  # alpha, beta (or a, b, s, h)
    sigma: float = np.nan  # residual standard deviation, n - k degrees of freedom
    n_obs: int = 0

    @property
    def ok(self) -> bool:
        return self.model == "market_adjusted" or bool(self.params)


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """OLS of y on [1, x]; returns (coefficients, residual std with n - k dof)."""
    design = np.column_stack([np.ones(len(y)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    dof = len(y) - design.shape[1]
    sigma = float(np.sqrt(resid @ resid / dof)) if dof > 0 else np.nan
    return coef, sigma


def fit_market_model(stock_ret: pd.Series, market_ret: pd.Series,
                     min_obs: int = MIN_FIT_OBS) -> Fit:
    """OLS market model on the dates where both returns exist."""
    both = pd.concat([stock_ret, market_ret], axis=1, join="inner").dropna()
    n = len(both)
    if n < max(min_obs, 3):
        return Fit("market", n_obs=n)
    coef, sigma = _ols(both.iloc[:, 0].to_numpy(float), both.iloc[:, 1].to_numpy(float))
    return Fit("market", {"alpha": float(coef[0]), "beta": float(coef[1])}, sigma, n)


def fit_ff3(stock_ret: pd.Series, factors: pd.DataFrame, min_obs: int = MIN_FIT_OBS) -> Fit:
    """OLS Fama-French 3-factor model of the excess return (factors in decimals)."""
    df = pd.concat([stock_ret.rename("ret"), factors[list(FF3_COLUMNS)]], axis=1,
                   join="inner").dropna()
    n = len(df)
    if n < max(min_obs, 5):
        return Fit("ff3", n_obs=n)
    y = (df["ret"] - df["rf"]).to_numpy(float)
    coef, sigma = _ols(y, df[["mkt_rf", "smb", "hml"]].to_numpy(float))
    params = dict(zip(("a", "b", "s", "h"), map(float, coef), strict=True))
    return Fit("ff3", params, sigma, n)


def expected_returns(fit: Fit, market_ret: pd.Series,
                     factors: pd.DataFrame | None = None) -> pd.Series:
    """Normal (expected) total return implied by `fit`, on the index of `market_ret`.

    For 'ff3' only the index of `market_ret` is used; the values come from `factors`.
    """
    if fit.model == "market_adjusted":
        return market_ret.astype(float)
    if not fit.ok:
        return pd.Series(np.nan, index=market_ret.index)
    if fit.model == "market":
        return fit.params["alpha"] + fit.params["beta"] * market_ret.astype(float)
    if fit.model == "ff3":
        if factors is None:
            raise ValueError("ff3 needs the factors table")
        f = factors.reindex(market_ret.index)
        p = fit.params
        return f["rf"] + p["a"] + p["b"] * f["mkt_rf"] + p["s"] * f["smb"] + p["h"] * f["hml"]
    raise ValueError(f"unknown model {fit.model!r}; choose from {MODELS}")


# ---- one event ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class EventAR:
    """Abnormal returns of one event in event time."""

    fit: Fit
    frame: pd.DataFrame  # index k (relative session); columns date, ret, ret_m, expected, ar

    def car(self, lo: int, hi: int) -> float:
        return car(self.frame["ar"], lo, hi)


def car(ar: pd.Series, lo: int, hi: int) -> float:
    """Sum of AR over relative days lo..hi; NaN if a day is missing or has no AR."""
    if lo > hi:
        raise ValueError(f"empty window [{lo}, {hi}]")
    days = range(lo, hi + 1)
    if not all(k in ar.index for k in days):
        return np.nan
    values = ar.loc[list(days)]
    return float(values.sum()) if values.notna().all() else np.nan


def _relative_index(sessions: pd.DatetimeIndex, j0: pd.Timestamp) -> int:
    pos = sessions.searchsorted(j0)
    if pos >= len(sessions) or sessions[pos] != j0:
        raise ValueError(f"J0 {j0.date()} is not one of the given sessions")
    return int(pos)


def event_abnormal_returns(
    stock_ret: pd.Series,
    market_ret: pd.Series,
    j0,
    *,
    sessions: pd.DatetimeIndex | None = None,
    model: str = "market",
    estimation_window: tuple[int, int] = (-250, -30),
    event_window: tuple[int, int] = (-10, 10),
    factors: pd.DataFrame | None = None,
    min_obs: int = MIN_FIT_OBS,
) -> EventAR:
    """Fit the normal-return model before J0 and return the ARs of the event window.

    `stock_ret` and `market_ret` are simple daily returns indexed by date. `sessions` is the
    ordered list of trading sessions defining event time (default: the dates of `stock_ret`).
    Sessions outside the available history are left out of the frame, so a window that runs
    past the data gives a missing CAR.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
    j0 = pd.Timestamp(j0).normalize()
    stock_ret = stock_ret.sort_index()
    if sessions is None:
        sessions = pd.DatetimeIndex(stock_ret.index)
    sessions = pd.DatetimeIndex(sessions).sort_values()
    p0 = _relative_index(sessions, j0)

    def span(lo: int, hi: int) -> pd.DatetimeIndex:
        return sessions[max(p0 + lo, 0): max(p0 + hi + 1, 0)]

    est_days = span(*estimation_window)
    if model == "market":
        fit = fit_market_model(stock_ret.reindex(est_days), market_ret.reindex(est_days), min_obs)
    elif model == "ff3":
        if factors is None:
            raise ValueError("model 'ff3' needs `factors` (columns mkt_rf, smb, hml, rf)")
        fit = fit_ff3(stock_ret.reindex(est_days), factors, min_obs)
    else:
        fit = Fit("market_adjusted", n_obs=0)

    ev_days = span(*event_window)
    k = np.arange(len(ev_days)) + (max(p0 + event_window[0], 0) - p0)
    rm = market_ret.reindex(ev_days)
    expected = expected_returns(fit, rm, factors)  # ff3 uses only the index of rm
    r = stock_ret.reindex(ev_days)
    frame = pd.DataFrame({
        "date": ev_days,
        "ret": r.to_numpy(float),
        "ret_m": rm.to_numpy(float),
        "expected": expected.to_numpy(float),
    }, index=pd.Index(k, name="k"))
    frame["ar"] = frame["ret"] - frame["expected"]
    return EventAR(fit, frame)


# ---- many events -------------------------------------------------------------------------------

def compute_cars(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    market: pd.DataFrame,
    company_index: Mapping[str, str],
    *,
    model: str = "market",
    windows: Mapping[str, tuple[int, int]] | None = None,
    estimation_window: tuple[int, int] | None = None,
    sessions_for: Callable[[str], pd.DatetimeIndex] | None = None,
    factors_for: Callable[[str], pd.DataFrame] | None = None,
    min_obs: int = MIN_FIT_OBS,
) -> pd.DataFrame:
    """CARs for every row of `events` (needs columns event_id, company_id, j0).

    `prices` and `market` follow the frozen `prices` and `market` schemas. `company_index`
    maps company_id to its index_id (universe.yaml `market_index`). `sessions_for(company_id)`
    returns the trading sessions of the company's exchange; default: its price dates.
    `factors_for(company_id)` returns the FF3 factors to use (needed for model 'ff3').

    Returns one row per event: event_id, one column per window, plus alpha, beta, sigma and
    n_est (estimation days actually used). The columns alpha and beta are NaN for 'ff3'.
    """
    windows = dict(windows or DEFAULT_WINDOWS)
    if estimation_window is None:
        estimation_window = tuple(load_params()["estimation_window"])
    lo = min(w[0] for w in windows.values())
    hi = max(w[1] for w in windows.values())

    stock = {cid: g.set_index("date")["ret"].sort_index()
             for cid, g in prices.groupby("company_id")}
    index_ret = {iid: g.set_index("date")["ret"].sort_index()
                 for iid, g in market.groupby("index_id")}

    rows = []
    for ev in events.itertuples(index=False):
        row: dict = {"event_id": ev.event_id}
        cid = ev.company_id
        try:
            if cid not in stock:
                raise KeyError(f"no prices for company {cid!r}")
            iid = company_index.get(cid)
            if iid not in index_ret:
                raise KeyError(f"no market index for company {cid!r} (index {iid!r})")
            res = event_abnormal_returns(
                stock[cid], index_ret[iid], ev.j0,
                sessions=sessions_for(cid) if sessions_for else None,
                model=model, estimation_window=estimation_window, event_window=(lo, hi),
                factors=factors_for(cid) if factors_for else None, min_obs=min_obs,
            )
            row.update({name: res.car(*w) for name, w in windows.items()})
            row.update(alpha=res.fit.params.get("alpha", np.nan),
                       beta=res.fit.params.get("beta", np.nan),
                       sigma=res.fit.sigma, n_est=res.fit.n_obs)
        except (KeyError, ValueError) as exc:
            log.warning("event %s: %s", ev.event_id, exc)
            row.update({name: np.nan for name in windows})
            row.update(alpha=np.nan, beta=np.nan, sigma=np.nan, n_est=0)
        rows.append(row)
    out = pd.DataFrame(rows, columns=["event_id", *windows, "alpha", "beta", "sigma", "n_est"])
    out["n_est"] = out["n_est"].astype(int)
    return out


# ---- Fama-French factors -----------------------------------------------------------------------

def read_ff3_csv(path: str | Path) -> pd.DataFrame:
    """Parse a daily 3-factor CSV from the Kenneth French data library.

    The files have a text preamble, a header row starting with a comma, values in percent and
    dates as YYYYMMDD; an annual block may follow. Returns columns mkt_rf, smb, hml, rf in
    decimals, indexed by date.
    """
    lines = Path(path).read_text(encoding="latin-1").splitlines()
    start = next(i for i, line in enumerate(lines) if line.replace(" ", "").lower().startswith(
        ",mkt-rf"))
    records = []
    for line in lines[start + 1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5 or not (parts[0].isdigit() and len(parts[0]) == 8):
            if records:
                break
            continue
        records.append(parts[:5])
    df = pd.DataFrame(records, columns=["date", *FF3_COLUMNS])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").astype(float) / 100.0
    return df


# ---- command: diagnostic -----------------------------------------------------------------------

def company_index_map(companies: list[str] | None = None) -> dict[str, str]:
    """company_id -> market index from config/universe.yaml."""
    with open(REPO_ROOT / "config" / "universe.yaml", encoding="utf-8") as f:
        universe = yaml.safe_load(f) or {}
    mapping = {c["company_id"]: c["market_index"] for c in universe.get("companies") or []}
    return {c: mapping[c] for c in companies if c in mapping} if companies else mapping


def _best_correlated_index(prices: pd.DataFrame, market: pd.DataFrame) -> dict[str, str]:
    """Fallback for companies missing from universe.yaml (synthetic fixtures)."""
    wide_m = market.pivot(index="date", columns="index_id", values="ret")
    out = {}
    for cid, g in prices.groupby("company_id"):
        corr = wide_m.corrwith(g.set_index("date")["ret"])
        out[cid] = str(corr.idxmax())
    return out


def main(argv: list[str] | None = None) -> int:
    from kpr.utils.io import read_table

    parser = argparse.ArgumentParser(
        prog="python -m kpr.build.returns",
        description="Recompute the CARs of the events table from prices and summarise them "
                    "(diagnostic, writes no shared table).",
    )
    parser.add_argument("--model", choices=[m for m in MODELS if m != "ff3"], default="market")
    parser.add_argument("--out", type=Path, help="optional CSV with one row per event")
    args = parser.parse_args(argv)

    events, prices, market = read_table("events"), read_table("prices"), read_table("market")
    mapping = company_index_map(sorted(events.company_id.unique()))
    missing = sorted(set(events.company_id) - set(mapping))
    if missing:
        guessed = _best_correlated_index(prices[prices.company_id.isin(missing)], market)
        log.warning("not in universe.yaml, index guessed by correlation: %s", guessed)
        mapping.update(guessed)

    cars = compute_cars(events, prices, market, mapping, model=args.model)
    merged = events[["event_id", "sign", "car_0_2"]].merge(cars, on="event_id",
                                                            suffixes=("_stored", ""))
    ok = merged.dropna(subset=["car_0_2", "car_0_2_stored"])
    log.info("model=%s events=%d with CAR=%d median n_est=%d", args.model, len(merged), len(ok),
             int(merged.n_est.median()) if len(merged) else 0)
    if len(ok) > 2:
        log.info("Spearman(recomputed, stored car_0_2) = %.3f",
                 ok["car_0_2"].corr(ok["car_0_2_stored"], method="spearman"))
    print(merged.groupby("sign")[["car_0_2", "car_m5_m1"]].agg(["count", "mean"]).to_string())
    if args.out:
        merged.to_csv(args.out, index=False)
        log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
