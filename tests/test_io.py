"""Frozen: shared I/O rules (fixture mode, writer ownership, atomic writes, data dir)."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from kpr.utils import io


@pytest.fixture
def data_dir(monkeypatch):
    # A system temp folder, guaranteed to be outside the git checkout.
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("KPR_DATA_DIR", tmp)
        monkeypatch.setenv("KPR_USE_FIXTURES", "0")
        yield Path(tmp)


def test_fixture_mode_reads_fixtures(monkeypatch):
    monkeypatch.setenv("KPR_USE_FIXTURES", "1")
    assert len(io.read_table("events")) > 0


def test_producer_can_write_and_read_back(data_dir, monkeypatch):
    monkeypatch.setenv("KPR_SESSION", "data")
    monkeypatch.setenv("KPR_USE_FIXTURES", "1")
    vix = io.read_table("vix")
    monkeypatch.setenv("KPR_USE_FIXTURES", "0")
    path = io.write_table(vix, "vix")
    assert path == data_dir / "interim" / "vix.parquet"
    pd.testing.assert_frame_equal(io.read_table("vix"), vix)
    assert not list(path.parent.glob("*.tmp"))


def test_other_session_cannot_write(data_dir, monkeypatch):
    monkeypatch.setenv("KPR_SESSION", "ml")
    monkeypatch.setenv("KPR_USE_FIXTURES", "1")
    vix = io.read_table("vix")
    with pytest.raises(PermissionError):
        io.write_table(vix, "vix")


def test_invalid_table_is_not_written(data_dir, monkeypatch):
    monkeypatch.setenv("KPR_SESSION", "data")
    with pytest.raises(ValueError):
        io.write_table(pd.DataFrame({"date": [1]}), "vix")
    assert not (data_dir / "interim" / "vix.parquet").exists()


def test_data_dir_must_be_outside_repo(monkeypatch):
    monkeypatch.setenv("KPR_DATA_DIR", str(io.REPO_ROOT / "data"))
    with pytest.raises(RuntimeError, match="outside"):
        io.data_dir()


def test_session_detection_from_env(monkeypatch):
    monkeypatch.setenv("KPR_SESSION", "Quant")
    assert io.current_session() == "quant"
