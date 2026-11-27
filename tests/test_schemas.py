"""Frozen: every fixture matches its contract, and validate() catches common mistakes."""

import pandas as pd
import pytest

from kpr.schemas import TABLES, SchemaError, validate
from kpr.utils.io import FIXTURES_DIR


@pytest.mark.parametrize("name", sorted(TABLES))
def test_fixture_matches_contract(name):
    path = FIXTURES_DIR / TABLES[name].filename
    assert path.exists(), f"missing fixture {path.name}: run python scripts/make_fixtures.py"
    validate(pd.read_parquet(path), name)


def test_missing_column_is_rejected():
    df = pd.read_parquet(FIXTURES_DIR / "vix.parquet").drop(columns="vix")
    with pytest.raises(SchemaError, match="missing columns"):
        validate(df, "vix")


def test_duplicate_keys_are_rejected():
    df = pd.read_parquet(FIXTURES_DIR / "vix.parquet")
    with pytest.raises(SchemaError, match="duplicate keys"):
        validate(pd.concat([df, df.head(1)]), "vix")


def test_bad_sign_is_rejected():
    df = pd.read_parquet(FIXTURES_DIR / "events.parquet")
    df.loc[0, "sign"] = "bad"
    with pytest.raises(SchemaError, match="sign must be"):
        validate(df, "events")


def test_extra_columns_only_where_allowed():
    vix = pd.read_parquet(FIXTURES_DIR / "vix.parquet").assign(extra=1.0)
    with pytest.raises(SchemaError, match="undeclared"):
        validate(vix, "vix")
    features = pd.read_parquet(FIXTURES_DIR / "features.parquet").assign(new_feature=1.0)
    validate(features, "features")
