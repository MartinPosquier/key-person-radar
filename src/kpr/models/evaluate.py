"""Metrics and bootstrap confidence intervals.

Owner: ml session. Task M4 in PARALLEL_SESSIONS.md §7. Dossier §9.7.

Regression (signed or absolute CAR): MAE, RMSE, out-of-sample R², Spearman correlation, and
the sign hit rate on non-neutral shocks. Classification (big move): PR-AUC, F1, recall
(no accuracy: the classes are imbalanced). Every metric comes with a percentile bootstrap
confidence interval over resampled events (1,000 resamples by default), and the gain of a
model over a baseline (B1 for H1) is tested by a paired bootstrap of the loss difference.

Run on a predictions file (columns y_true, y_pred, optional y_base and sign):
    python -m kpr.models.evaluate predictions.csv --task regression
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, recall_score

from kpr.utils.logging import get_logger

log = get_logger(__name__)

N_BOOT = 1000  # dossier §9.7
ALPHA = 0.05  # 95% intervals
SEED = 42  # dossier §13.4


# ---- point metrics --------------------------------------------------------------------------


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def r2_oos(y, p, benchmark: float = 0.0) -> float:
    """Out-of-sample R²: 1 - SSE(model) / SSE(constant benchmark).

    The benchmark is a value known before the test period, typically the training mean
    (Campbell-Thompson). With benchmark = 0, R² > 0 means beating "no abnormal return".
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    sst = np.sum((y - benchmark) ** 2)
    return float(1 - np.sum((y - p) ** 2) / sst) if sst > 0 else np.nan


def spearman(y, p) -> float:
    if np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    return float(spearmanr(y, p).statistic)


def sign_hit_rate(y, p) -> float:
    """Share of events whose predicted sign is right (pass non-neutral shocks only)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.sign(y) == np.sign(p))) if len(y) else np.nan


def pr_auc(y, p) -> float:
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else np.nan


def f1(y, p, threshold: float = 0.5) -> float:
    return float(f1_score(y, np.asarray(p) >= threshold, zero_division=0))


def recall(y, p, threshold: float = 0.5) -> float:
    return float(recall_score(y, np.asarray(p) >= threshold, zero_division=0))


REGRESSION: dict[str, Callable] = {
    "mae": mae, "rmse": rmse, "r2_oos": r2_oos, "spearman": spearman,
}
CLASSIFICATION: dict[str, Callable] = {"pr_auc": pr_auc, "f1": f1, "recall": recall}


# ---- bootstrap ------------------------------------------------------------------------------


def _boot_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_ci(y, p, metric: Callable, n_boot: int = N_BOOT, alpha: float = ALPHA,
                 seed: int = SEED) -> tuple[float, float, float]:
    """Point value and percentile bootstrap interval of metric(y, p), resampling events."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    point = metric(y, p)
    if len(y) < 2:
        return point, np.nan, np.nan
    stats = np.array([metric(y[i], p[i]) for i in _boot_indices(len(y), n_boot, seed)])
    stats = stats[~np.isnan(stats)]
    if not len(stats):
        return point, np.nan, np.nan
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def evaluate(y, p, task: str = "regression", sign=None, benchmark: float = 0.0,
             n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """All metrics of §9.7 for one set of predictions, each with its bootstrap interval.

    `sign` (optional, from the events table) restricts the sign hit rate to non-neutral
    shocks; without it, the hit rate uses every event.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    if task == "regression":
        metrics = dict(REGRESSION, r2_oos=lambda a, b: r2_oos(a, b, benchmark))
    elif task == "classification":
        metrics = CLASSIFICATION
    else:
        raise ValueError(f"task must be 'regression' or 'classification', got {task!r}")
    rows = []
    for name, fn in metrics.items():
        point, lo, hi = bootstrap_ci(y, p, fn, n_boot=n_boot, seed=seed)
        rows.append({"metric": name, "value": point, "lo": lo, "hi": hi, "n": len(y)})
    if task == "regression":
        mask = np.ones(len(y), bool) if sign is None else np.asarray(sign) != "neutral"
        point, lo, hi = bootstrap_ci(y[mask], p[mask], sign_hit_rate, n_boot=n_boot, seed=seed)
        rows.append({"metric": "sign_hit_rate", "value": point, "lo": lo, "hi": hi,
                     "n": int(mask.sum())})
    return pd.DataFrame(rows)


def paired_bootstrap_gain(y, p_model, p_base, loss: str = "abs", n_boot: int = N_BOOT,
                          alpha: float = ALPHA, seed: int = SEED) -> dict[str, float]:
    """Gain of a model over a baseline: mean loss(base) - mean loss(model), paired bootstrap.

    Positive gain = the model has smaller errors. `p_value` is the one-sided share of
    resamples where the gain is <= 0 (H0: the model is no better than the baseline).
    """
    y, pm, pb = (np.asarray(a, float) for a in (y, p_model, p_base))
    if loss == "abs":
        diff = np.abs(y - pb) - np.abs(y - pm)
    elif loss == "squared":
        diff = (y - pb) ** 2 - (y - pm) ** 2
    else:
        raise ValueError(f"loss must be 'abs' or 'squared', got {loss!r}")
    boots = diff[_boot_indices(len(diff), n_boot, seed)].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"gain": float(diff.mean()), "lo": float(lo), "hi": float(hi),
            "p_value": float(np.mean(boots <= 0)), "n": len(diff)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Metrics with bootstrap CIs (dossier §9.7)")
    parser.add_argument("predictions", help="CSV or Parquet with y_true, y_pred[, y_base, sign]")
    parser.add_argument("--task", choices=["regression", "classification"],
                        default="regression")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)

    path = args.predictions
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    table = evaluate(df["y_true"], df["y_pred"], args.task, sign=df.get("sign"),
                     n_boot=args.n_boot)
    log.info("metrics (95%% bootstrap intervals):\n%s", table.round(5).to_string(index=False))
    if "y_base" in df.columns and args.task == "regression":
        gain = paired_bootstrap_gain(df["y_true"], df["y_pred"], df["y_base"],
                                     n_boot=args.n_boot)
        log.info("gain over baseline (MAE): %s", {k: round(v, 6) for k, v in gain.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
