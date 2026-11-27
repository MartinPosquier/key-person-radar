"""Chronological train/validation/test split with gap; reserved natural experiments.

Owner: quant session. Task Q6 in PARALLEL_SESSIONS.md §7. Dossier §9.5.

``period_split`` maps each event to its period by J0 (Q5): train 2015-2021, validation
2022-2023, test 2024-2025, and "reserved" for natural experiments. Events before the train
period count as train (they rarely survive the 120-session estimation filter anyway).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def period_split(j0: pd.Series, splits: Mapping, reserved: pd.Series | None = None) -> pd.Series:
    """train / validation / test by J0 (the `splits` block of quant.yaml); reserved first."""
    j0 = pd.to_datetime(j0)
    train_end = pd.Timestamp(splits["train"][1])
    val_start, val_end = map(pd.Timestamp, splits["validation"])
    test_start, test_end = map(pd.Timestamp, splits["test"])
    if not train_end < val_start <= val_end < test_start <= test_end:
        raise ValueError(f"split periods overlap or are out of order: {dict(splits)}")
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


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("kpr.build.splits: quant session, task Q6")


if __name__ == "__main__":
    raise SystemExit(main())
