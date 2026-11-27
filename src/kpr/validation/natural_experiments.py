"""Ex ante score vs real reaction on reserved natural experiments.

Owner: quant session. Task Q10 in PARALLEL_SESSIONS.md §7. Dossier §11.2.

The candidate list lives in `config/natural_experiments.csv` (task Q3). `load_candidates`
serves events.py (Q5), which flags those events as reserved; the validation itself (`main`)
comes with Q10.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
CANDIDATES_PATH = REPO_ROOT / "config" / "natural_experiments.csv"

COLUMNS = ("event_id", "person_name", "company_id", "date", "kind", "wanted_by_company",
           "source_url", "notes")
KINDS = (
    "arrest",                # arrest or criminal charge of the person
    "death",                 # death in office
    "health",                # illness or emergency treatment disclosed
    "legal_action",          # regulator or court action against the person
    "resignation_personal",  # resignation for a personal reason (conduct, private life)
    "dismissal",             # sudden removal by the board
    "surprise_departure",    # sudden departure not explained by conduct or health
    "surprise_appointment",  # unexpected arrival of a new leader
)
# Days around a natural experiment kept out of training and calibration (dossier §11.2).
EXCLUSION_DAYS = 10


def load_candidates(path: str | Path = CANDIDATES_PATH) -> pd.DataFrame:
    """Read and check the candidate list; `date` is the UTC day the news broke."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(df.columns) != COLUMNS:
        raise ValueError(f"{path}: columns must be {COLUMNS}, got {tuple(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    bad_kind = sorted(set(df["kind"]) - set(KINDS))
    if bad_kind:
        raise ValueError(f"{path}: unknown kind {bad_kind}; allowed: {KINDS}")
    flags = df["wanted_by_company"].str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError(f"{path}: wanted_by_company must be true or false")
    df["wanted_by_company"] = flags.eq("true")
    if df["event_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate event_id")
    return df


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("kpr.validation.natural_experiments: quant session, task Q10")


if __name__ == "__main__":
    raise SystemExit(main())
