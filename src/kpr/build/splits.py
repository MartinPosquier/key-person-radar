"""Chronological train/validation/test split with gap; reserved natural experiments.

Owner: quant session. Task Q6 in PARALLEL_SESSIONS.md §7. Dossier §9.5.

- ``period_split`` maps each event to its period by J0: train up to 2021 (2015-2021, plus
  the rare earlier events), validation 2022-2023, test 2024-2025, and "reserved" for natural
  experiments, whatever their date.
- ``boundary_gap`` marks the events whose J0 falls in the last `gap_sessions` business days
  before the start of validation or test. Their reaction window can run into the next
  period, so they would leak across the boundary (§9.5: "un trou de 10 séances").
- ``assign_splits`` applies both to an events table. The frozen schema has no "gap" split,
  so gap events keep their period and are flagged ``excluded`` with reason ``split_gap``
  (appended to any other reason): no model trains or is scored on them, and they can still
  be counted.

Inside the training period, the expanding-window cross-validation and its own gap are the
ml session's (``kpr.models.cv.ExpandingYearSplit``), with the same business-day convention.

``python -m kpr.build.splits`` prints the split of the events table (diagnostic).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping

import numpy as np
import pandas as pd

GAP_REASON = "split_gap"


def _periods(splits: Mapping) -> tuple[pd.Timestamp, ...]:
    train_end = pd.Timestamp(splits["train"][1])
    val_start, val_end = map(pd.Timestamp, splits["validation"])
    test_start, test_end = map(pd.Timestamp, splits["test"])
    if not train_end < val_start <= val_end < test_start <= test_end:
        raise ValueError(f"split periods overlap or are out of order: {dict(splits)}")
    return train_end, val_start, val_end, test_start, test_end


def period_split(j0: pd.Series, splits: Mapping, reserved: pd.Series | None = None) -> pd.Series:
    """train / validation / test by J0 (the `splits` block of quant.yaml); reserved first."""
    j0 = pd.to_datetime(j0)
    train_end, val_start, val_end, test_start, test_end = _periods(splits)
    out = np.select(
        [j0 <= train_end, (j0 >= val_start) & (j0 <= val_end),
         (j0 >= test_start) & (j0 <= test_end)],
        ["train", "validation", "test"], default="",
    )
    out = pd.Series(out, index=j0.index, name="split", dtype=object)
    if reserved is not None:
        out[reserved.astype(bool).to_numpy()] = "reserved"
    unassigned = out == ""
    if unassigned.any():
        raise ValueError(f"{int(unassigned.sum())} events fall between or after the periods")
    return out


def boundary_gap(j0: pd.Series, splits: Mapping, gap_sessions: int) -> pd.Series:
    """True if J0 is in the `gap_sessions` business days before validation or test starts."""
    j0 = pd.to_datetime(j0)
    _, val_start, _, test_start, _ = _periods(splits)
    out = pd.Series(False, index=j0.index, name="in_gap")
    if gap_sessions <= 0:
        return out
    for start in (val_start, test_start):
        cutoff = start - pd.offsets.BDay(gap_sessions)
        out |= (j0 >= cutoff) & (j0 < start)
    return out


def assign_splits(events: pd.DataFrame, splits: Mapping, gap_sessions: int) -> pd.DataFrame:
    """Set `split` and flag boundary-gap events as excluded (`split_gap`) on a copy."""
    ev = events.copy()
    when = ev["j0"].fillna(ev["date_utc"])
    reserved = ev["is_natural_experiment"].astype(bool)
    ev["split"] = period_split(when, splits, reserved)
    gap = boundary_gap(when, splits, gap_sessions) & ~reserved
    reasons = ev["exclusion_reason"].astype(object)
    ev["exclusion_reason"] = [
        (f"{r};{GAP_REASON}" if isinstance(r, str) and r else GAP_REASON) if g else r
        for r, g in zip(reasons, gap, strict=True)
    ]
    ev["exclusion_reason"] = ev["exclusion_reason"].astype(object)
    ev["excluded"] = ev["excluded"].astype(bool) | gap.to_numpy()
    return ev


def summary(events: pd.DataFrame) -> pd.DataFrame:
    """Events per split: all, kept, gap-excluded, and kept by sign."""
    gap = events["exclusion_reason"].fillna("").str.contains(GAP_REASON)
    kept = events[~events["excluded"]]
    table = pd.DataFrame({
        "events": events.groupby("split").size(),
        "kept": kept.groupby("split").size(),
        "split_gap": events[gap].groupby("split").size(),
    })
    by_sign = kept.groupby(["split", "sign"]).size().unstack(fill_value=0)
    return table.join(by_sign).fillna(0).astype(int)


def main(argv: list[str] | None = None) -> int:
    from kpr.utils.io import read_table

    parser = argparse.ArgumentParser(
        prog="python -m kpr.build.splits",
        description="Print the train / validation / test / reserved split of the events "
                    "table (written by kpr.build.events).",
    )
    parser.parse_args(argv)
    print(summary(read_table("events")).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
