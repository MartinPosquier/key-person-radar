"""Time-series cross-validation harness: expanding yearly window with a gap (dossier §9.5).

Owner: ml session. Task M3 in PARALLEL_SESSIONS.md §7.

Fold k trains on every event up to the end of year `first_train_end_year + k - 1` and
validates on the next calendar year. Training events whose J0 falls within `gap_sessions`
trading sessions before the validation year are dropped, so that a shock straddling the
boundary (its reaction window runs to J0+2) cannot leak. The splitter follows the
scikit-learn CV protocol, so it plugs into `cross_validate`, `GridSearchCV`, `RFECV`, ...

The test split is never touched here: `modelling_frame` refuses it unless the caller
passes `allow_test=True`, which only the final evaluation (task M12) may do.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
TARGETS = ("car_signed", "car_abs", "big_move")
SESSIONS_PER_WINDOW = 3  # CAR[0, 2] spans 3 sessions


def load_ml_params() -> dict:
    return yaml.safe_load((CONFIG_DIR / "params" / "ml.yaml").read_text(encoding="utf-8"))


class ExpandingYearSplit:
    """Expanding-window yearly splitter with a gap of `gap_sessions` business days.

    `dates` must be aligned with the rows of the X given to `split`.
    """

    def __init__(self, dates, first_train_end_year: int, gap_sessions: int = 10,
                 last_val_year: int | None = None):
        self.dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
        self.first_train_end_year = first_train_end_year
        self.gap_sessions = gap_sessions
        self.last_val_year = last_val_year

    def _val_years(self) -> list[int]:
        last = self.last_val_year or int(self.dates.year.max())
        return [y for y in range(self.first_train_end_year + 1, last + 1)
                if (self.dates.year == y).any()]

    def split(self, X=None, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if X is not None and len(X) != len(self.dates):
            raise ValueError(f"X has {len(X)} rows but the splitter has {len(self.dates)} dates")
        for year in self._val_years():
            val_start = pd.Timestamp(year=year, month=1, day=1)
            cutoff = val_start - pd.offsets.BDay(self.gap_sessions)
            train = np.flatnonzero(self.dates < cutoff)
            val = np.flatnonzero(self.dates.year == year)
            if len(train) and len(val):
                yield train, val

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return sum(1 for _ in self.split())


def add_targets(events: pd.DataFrame, features: pd.DataFrame | None = None,
                big_move_sd: float = 2.0) -> pd.DataFrame:
    """The three targets of §9.1.

    car_signed: CAR[0,2]; car_abs: |CAR[0,2]|; big_move: |CAR[0,2]| above `big_move_sd`
    standard deviations of the usual 3-session volatility (vol_20d * sqrt(3), from the
    features table; NaN when the volatility is unknown).
    """
    out = events.copy()
    out["car_signed"] = out["car_0_2"]
    out["car_abs"] = out["car_0_2"].abs()
    out["big_move"] = np.nan
    if features is not None and "vol_20d" in features.columns:
        vol = out["event_id"].map(features.set_index("event_id")["vol_20d"])
        threshold = big_move_sd * vol * np.sqrt(SESSIONS_PER_WINDOW)
        out["big_move"] = (out["car_abs"] > threshold).astype(float).where(threshold.notna())
    return out


def modelling_frame(events: pd.DataFrame, features: pd.DataFrame,
                    splits: tuple[str, ...] = ("train",), allow_test: bool = False
                    ) -> pd.DataFrame:
    """Usable events of the given splits joined to their features, sorted by J0.

    Drops excluded events and natural experiments (reserved for validation, §11.2) and
    events without a CAR[0,2].
    """
    if "test" in splits and not allow_test:
        raise PermissionError("the test split is used once, in task M12 (dossier §9.5)")
    usable = events[
        events["split"].isin(splits)
        & ~events["excluded"].astype(bool)
        & ~events["is_natural_experiment"].astype(bool)
        & events["car_0_2"].notna()
    ]
    usable = add_targets(usable, features)
    frame = usable.merge(features, on="event_id", how="inner", suffixes=("", "_feat"))
    return frame.sort_values(["j0", "event_id"]).reset_index(drop=True)


def fit_predict(model, X_train, y_train, X_new, proba: bool = False) -> np.ndarray:
    """Fit a clone of `model` and predict; for classifiers, the probability of class 1.

    A training fold with a single class gets a constant prediction (that class) instead of
    a fit, which most classifiers refuse.
    """
    if proba and pd.Series(y_train).nunique() < 2:
        return np.full(len(X_new), float(pd.Series(y_train).iloc[0] == 1))
    est = clone(model).fit(X_train, y_train)
    return est.predict_proba(X_new)[:, 1] if proba else est.predict(X_new)


def oof_predictions(model, X: pd.DataFrame, y: pd.Series, splitter,
                    ids: pd.Series | None = None, proba: bool = False) -> pd.DataFrame:
    """Fit a fresh clone of `model` on each training fold; collect validation predictions.

    Returns one row per validated event: id, fold, y_true, y_pred. Preprocessing (imputing,
    scaling, selection) must live inside `model` so that it is fitted on the fold only.
    """
    ids = pd.Series(np.arange(len(X))) if ids is None else ids.reset_index(drop=True)
    parts = []
    for fold, (tr, va) in enumerate(splitter.split(X)):
        pred = fit_predict(model, X.iloc[tr], y.iloc[tr], X.iloc[va], proba)
        parts.append(pd.DataFrame({"id": ids.iloc[va].to_numpy(), "fold": fold,
                                   "y_true": y.iloc[va].to_numpy(), "y_pred": pred}))
    cols = ["id", "fold", "y_true", "y_pred"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
