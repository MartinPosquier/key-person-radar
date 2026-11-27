"""Owner: quant session. Checks of config/natural_experiments.csv (task Q3, dossier §11.2)."""

import pandas as pd
import pytest

from kpr.validation import natural_experiments as ne


@pytest.fixture(scope="module")
def candidates():
    return ne.load_candidates()


def test_at_least_15_candidates(candidates):
    assert len(candidates) >= 15


def test_every_row_is_complete(candidates):
    for col in ("event_id", "person_name", "company_id", "source_url", "notes"):
        assert (candidates[col].str.strip() != "").all(), col
    assert candidates["source_url"].str.startswith("https://").all()


def test_dates_cover_the_study_period(candidates):
    assert candidates["date"].min() >= pd.Timestamp("2014-01-01")
    assert candidates["date"].max() <= pd.Timestamp("2025-12-31")
    # Most cases must fall where a score of the previous year can exist (2015-2025).
    assert (candidates["date"] >= pd.Timestamp("2015-01-01")).sum() >= 15


def test_both_kinds_of_shocks_are_present(candidates):
    assert candidates["wanted_by_company"].any()
    assert (~candidates["wanted_by_company"]).sum() >= 5  # not wanted by the company
    assert candidates["kind"].nunique() >= 5


def test_loader_rejects_bad_files(tmp_path):
    good = ne.CANDIDATES_PATH.read_text(encoding="utf-8").splitlines()
    bad_kind = tmp_path / "bad_kind.csv"
    bad_kind.write_text("\n".join([good[0], good[1].replace(",health,", ",gossip,")]),
                        encoding="utf-8")
    with pytest.raises(ValueError, match="unknown kind"):
        ne.load_candidates(bad_kind)
    dup = tmp_path / "dup.csv"
    dup.write_text("\n".join([good[0], good[1], good[1]]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        ne.load_candidates(dup)
