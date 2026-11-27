"""Classic event study on the events table, before any machine learning.

Owner: quant session. Task Q7 in PARALLEL_SESSIONS.md §7. Dossier §7.6.
Used by notebooks/02_event_study.ipynb; also runnable: ``python -m kpr.validation.event_study``.

1. Mean CAR[0, 2] by sign (negative, neutral, positive).
2. Student t-test of mean CAR = 0, completed by a sign test (less sensitive to outliers).
3. Media versus control companies: is the mean |CAR| larger for media figures?
4. Mean cumulative abnormal return from J-10 to J+10 by sign, with a 95 % band.

Only kept events (not excluded) are used, and by default only the train and validation
splits: the test period stays untouched until the final evaluation (§9.5), and reserved
natural experiments are kept for §11.2.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from kpr.build import events as E
from kpr.build import returns as R
from kpr.utils.logging import get_logger

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPLITS = ("train", "validation")
SIGN_ORDER = ("negative", "neutral", "positive")


def select(events: pd.DataFrame, splits: Sequence[str] = DEFAULT_SPLITS) -> pd.DataFrame:
    """Kept events of the given splits."""
    return events[~events["excluded"].astype(bool) & events["split"].isin(splits)].copy()


def company_groups(path: Path | None = None) -> dict[str, str]:
    """company_id -> 'media' | 'control' from config/universe.yaml."""
    path = path or REPO_ROOT / "config" / "universe.yaml"
    with open(path, encoding="utf-8") as f:
        universe = yaml.safe_load(f) or {}
    return {c["company_id"]: c.get("group", "unknown") for c in universe.get("companies") or []}


# ---- tests on CARs -----------------------------------------------------------------------------

def mean_zero_tests(x: pd.Series) -> dict:
    """n, mean, median, std, Student t and p, sign test p (two-sided) for H0: CAR = 0."""
    x = pd.Series(x, dtype=float).dropna()
    n = len(x)
    out = {"n": n, "mean": x.mean() if n else np.nan, "median": x.median() if n else np.nan,
           "std": x.std() if n > 1 else np.nan, "t": np.nan, "p_t": np.nan,
           "share_pos": (x > 0).mean() if n else np.nan, "p_sign": np.nan}
    if n > 1 and x.std() > 0:
        t = stats.ttest_1samp(x, 0.0)
        out.update(t=float(t.statistic), p_t=float(t.pvalue))
    nonzero = x[x != 0]
    if len(nonzero):
        out["p_sign"] = float(stats.binomtest(int((nonzero > 0).sum()), len(nonzero)).pvalue)
    return out


def mean_car_table(events: pd.DataFrame, by: str = "sign", car: str = "car_0_2") -> pd.DataFrame:
    """One row per group of `by` with the statistics of ``mean_zero_tests``."""
    rows = {key: mean_zero_tests(g[car]) for key, g in events.groupby(by)}
    table = pd.DataFrame.from_dict(rows, orient="index")
    if by == "sign":
        table = table.reindex([s for s in SIGN_ORDER if s in table.index])
    table.index.name = by
    return table


def compare_groups(events: pd.DataFrame, groups: Mapping[str, str],
                   car: str = "car_0_2") -> dict:
    """Mean |CAR| of media versus control companies, one-sided tests (media larger)."""
    g = events["company_id"].map(groups).fillna("unknown")
    media = events.loc[g == "media", car].abs().dropna()
    control = events.loc[g == "control", car].abs().dropna()
    out = {"n_media": len(media), "n_control": len(control),
           "mean_abs_media": media.mean(), "mean_abs_control": control.mean(),
           "p_welch": np.nan, "p_mannwhitney": np.nan}
    if len(media) > 1 and len(control) > 1:
        out["p_welch"] = float(stats.ttest_ind(media, control, equal_var=False,
                                               alternative="greater").pvalue)
        out["p_mannwhitney"] = float(stats.mannwhitneyu(media, control,
                                                        alternative="greater").pvalue)
    return out


# ---- CAR curves --------------------------------------------------------------------------------

def car_paths(events: pd.DataFrame, prices: pd.DataFrame, market: pd.DataFrame,
              window: tuple[int, int] = (-10, 10), universe=None,
              params: Mapping | None = None) -> pd.DataFrame:
    """Cumulative abnormal return from J+window[0] to each day k, one row per event.

    Columns are k = window[0] .. window[1]; a row is NaN after the first missing AR.
    """
    params = params or E.load_params()
    markets = E.Markets(prices, market, universe)
    stock = {c: g.set_index("date")["ret"].sort_index() for c, g in prices.groupby("company_id")}
    est = tuple(params["returns"]["estimation_window"])
    ks = list(range(window[0], window[1] + 1))
    rows = {}
    for ev in events.itertuples(index=False):
        if ev.company_id not in stock or pd.isna(ev.j0):
            continue
        res = R.event_abnormal_returns(
            stock[ev.company_id], markets.index_ret(markets.index_of(ev.company_id)), ev.j0,
            sessions=markets.sessions(ev.company_id), estimation_window=est,
            event_window=window)
        ar = res.frame["ar"].reindex(ks)
        rows[ev.event_id] = ar.cumsum(skipna=False)
    return pd.DataFrame.from_dict(rows, orient="index", columns=ks)


def mean_paths(paths: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Mean path and 95 % band per label: columns (label, stat) with stat in mean/lo/hi/n."""
    out = {}
    for label, idx in labels.reindex(paths.index).groupby(labels).groups.items():
        p = paths.loc[idx]
        n = p.notna().sum()
        mean = p.mean()
        se = p.std() / np.sqrt(n.where(n > 1))
        out[(label, "mean")] = mean
        out[(label, "lo")] = mean - 1.96 * se
        out[(label, "hi")] = mean + 1.96 * se
        out[(label, "n")] = n
    return pd.DataFrame(out)


def plot_mean_paths(summary: pd.DataFrame, ax=None, title: str = "Mean CAR around J0"):
    """Line per label with its 95 % band; returns the matplotlib Axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"negative": "#c0392b", "neutral": "#7f8c8d", "positive": "#27ae60"}
    labels = [lab for lab in dict.fromkeys(summary.columns.get_level_values(0))]
    for label in labels:
        s = summary[label]
        color = colors.get(label)
        ax.plot(s.index, s["mean"] * 100, label=f"{label} (n={int(s['n'].max())})", color=color)
        ax.fill_between(s.index, s["lo"] * 100, s["hi"] * 100, alpha=0.15, color=color)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("sessions relative to J0")
    ax.set_ylabel("cumulative abnormal return (%)")
    ax.set_title(title)
    ax.legend()
    return ax


# ---- command -------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    from kpr.utils.io import read_table

    parser = argparse.ArgumentParser(prog="python -m kpr.validation.event_study",
                                     description="Event study of dossier §7.6 (tables only).")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    args = parser.parse_args(argv)

    events = select(read_table("events"), args.splits)
    log.info("kept events in %s: %d", args.splits, len(events))
    with pd.option_context("display.float_format", "{:.4f}".format):
        print("CAR[0,2] by sign\n", mean_car_table(events).to_string(), sep="")
        print("\nCAR[-5,-1] by sign (anticipation)\n",
              mean_car_table(events, car="car_m5_m1").to_string(), sep="")
        print("\nmedia vs control |CAR[0,2]|\n", compare_groups(events, company_groups()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
