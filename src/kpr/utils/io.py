"""Frozen I/O helpers: the only way sessions read and write shared tables.

- Paths are resolved under KPR_DATA_DIR (from the environment or the worktree's .env).
- `write_table` validates against `kpr.schemas`, refuses to write a table the current session
  does not produce, and writes atomically (temp file + rename) so that a session reading from
  another worktree never sees a half-written file.
- With KPR_USE_FIXTURES=1, `read_table` reads the synthetic fixtures in tests/fixtures/.

Change this file only through a contract PR (PARALLEL_SESSIONS.md §6).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from kpr.schemas import TABLES, validate

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

# Worktree folder name -> session. The main checkout may write anything (integration).
FOLDER_TO_SESSION = {
    "kpr-data": "data",
    "kpr-quant": "quant",
    "kpr-ml": "ml",
    "key-person-radar": "main",
}

load_dotenv(REPO_ROOT / ".env", override=False)


def current_session() -> str:
    """Return 'data', 'quant', 'ml' or 'main' (integration). Unknown folders count as 'main'."""
    forced = os.environ.get("KPR_SESSION", "").strip().lower()
    if forced:
        return forced
    return FOLDER_TO_SESSION.get(REPO_ROOT.name, "main")


def data_dir() -> Path:
    value = os.environ.get("KPR_DATA_DIR", "").strip()
    if not value:
        raise RuntimeError("KPR_DATA_DIR is not set: copy .env.example to .env and fill it in")
    path = Path(value).expanduser()
    if REPO_ROOT == path or REPO_ROOT in path.parents:
        raise RuntimeError("KPR_DATA_DIR must be outside the git checkout")
    return path


def use_fixtures() -> bool:
    return os.environ.get("KPR_USE_FIXTURES", "0").strip() == "1"


def table_path(name: str) -> Path:
    table = TABLES[name]
    return data_dir() / table.folder / table.filename


def raw_dir(source: str) -> Path:
    """Cache folder for raw API responses of `source` (data session only)."""
    path = data_dir() / "raw" / source
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_table(name: str, validate_schema: bool = True) -> pd.DataFrame:
    if name not in TABLES:
        raise KeyError(f"unknown table {name!r}")
    path = FIXTURES_DIR / TABLES[name].filename if use_fixtures() else table_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist yet. Is the producer's PR merged and run? "
            "Set KPR_USE_FIXTURES=1 to work on synthetic data meanwhile."
        )
    df = pd.read_parquet(path)
    return validate(df, name) if validate_schema else df


def _check_writer(name: str) -> None:
    session = current_session()
    producer = TABLES[name].producer
    if session not in ("main", producer):
        raise PermissionError(
            f"session {session!r} may not write table {name!r} (producer: {producer!r}). "
            "Write an inbox request instead (PARALLEL_SESSIONS.md §9)."
        )


def _atomic_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def write_table(df: pd.DataFrame, name: str) -> Path:
    """Validate and atomically write a shared table. Returns the written path."""
    if name not in TABLES:
        raise KeyError(f"unknown table {name!r}")
    _check_writer(name)
    validate(df, name)
    return _atomic_parquet(df, table_path(name))


def write_fixture(df: pd.DataFrame, name: str) -> Path:
    """Used only by scripts/make_fixtures.py."""
    validate(df, name)
    return _atomic_parquet(df, FIXTURES_DIR / TABLES[name].filename)
