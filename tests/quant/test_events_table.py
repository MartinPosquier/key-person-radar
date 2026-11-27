"""Owner: quant session. Tests of the events table (task Q5, dossier §7.7, §9.5, §11.2)."""

import pandas as pd
import pytest

from kpr.build import events as E
from kpr.build import splits as S
from kpr.schemas import validate

FIXTURES = E.REPO_ROOT / "tests" / "fixtures"
SPLITS = E.load_params()["splits"]


@pytest.fixture(scope="module")
def built():
    t = {n: pd.read_parquet(FIXTURES / f"{n}.parquet")
         for n in ("panel", "prices", "market", "earnings", "wiki_views", "gdelt_daily")}
    return E.build_events(**t, universe={})


def _candidates(rows):
    return pd.DataFrame(rows, columns=["event_id", "company_id", "date"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


# ---- period split (§9.5) -----------------------------------------------------------------------

def test_period_split_by_j0():
    j0 = pd.Series(pd.to_datetime(["2014-06-02", "2015-01-02", "2021-12-31", "2022-01-03",
                                   "2023-12-29", "2024-01-02", "2025-12-31"]))
    got = S.period_split(j0, SPLITS)
    assert list(got) == ["train", "train", "train", "validation", "validation", "test", "test"]


def test_reserved_overrides_the_period():
    j0 = pd.Series(pd.to_datetime(["2016-03-01", "2024-03-01"]))
    got = S.period_split(j0, SPLITS, reserved=pd.Series([False, True]))
    assert list(got) == ["train", "reserved"]


def test_period_split_rejects_dates_after_the_test_period():
    with pytest.raises(ValueError, match="fall between or after"):
        S.period_split(pd.Series(pd.to_datetime(["2026-02-02"])), SPLITS)


def test_period_split_rejects_overlapping_periods():
    bad = dict(SPLITS, validation=["2021-06-01", "2023-12-31"])
    with pytest.raises(ValueError, match="overlap"):
        S.period_split(pd.Series(pd.to_datetime(["2016-01-04"])), bad)


# ---- natural experiments (§11.2) ---------------------------------------------------------------

def test_natural_experiment_flags_company_neighbours(built):
    ev = built.iloc[0]
    cand = _candidates([["NE900", ev.company_id, ev.date_utc + pd.Timedelta(days=10)],
                        ["NE901", "OTHER", ev.date_utc]])
    flags = E.flag_natural_experiments(built, cand)
    assert flags.iloc[0]
    same_co = built["company_id"] == ev.company_id
    near = (built["date_utc"] - (ev.date_utc + pd.Timedelta(days=10))).abs().dt.days <= 10
    assert (flags == (same_co & near)).all()
    far = _candidates([["NE902", ev.company_id, ev.date_utc + pd.Timedelta(days=11)]])
    assert not E.flag_natural_experiments(built.iloc[[0]], far).iloc[0]


# ---- finalized table ---------------------------------------------------------------------------

def test_finalize_gives_a_valid_events_table(built):
    ev = built.iloc[3]
    cand = _candidates([["NE900", ev.company_id, ev.date_utc]])
    out = E.finalize(built, cand)
    validate(out, "events")
    assert list(out.columns) == E.EVENT_COLUMNS
    assert len(out) == len(built)
    row = out.set_index("event_id").loc[ev.event_id]
    assert row["is_natural_experiment"] and row["split"] == "reserved"
    assert set(out.loc[~out.is_natural_experiment, "split"]) == {"train"}  # fixtures: 2020-21
    assert out["date_utc"].is_monotonic_increasing


def test_finalize_with_the_real_candidate_list(built):
    from kpr.validation import natural_experiments as ne
    out = E.finalize(built, ne.load_candidates())  # fake companies: nothing reserved
    assert not out["is_natural_experiment"].any()


# ---- command -------------------------------------------------------------------------------

def test_parse_overrides():
    got = E._parse_overrides(["shocks.z_wiki_min=2.5", "filters.earnings_window_sessions=2"])
    assert got == {"shocks": {"z_wiki_min": 2.5}, "filters": {"earnings_window_sessions": 2}}
    with pytest.raises(ValueError):
        E._parse_overrides(["shocks.z_wiki_min"])


def test_main_on_fixtures_never_writes_the_shared_table(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("KPR_USE_FIXTURES", "1")

    def forbidden(*_a, **_k):
        raise AssertionError("must not write the shared table from fixtures")

    monkeypatch.setattr(E, "write_table", forbidden)
    assert E.main([]) == 0
    assert "train" in capsys.readouterr().out
    out = tmp_path / "events.parquet"
    assert E.main(["--out", str(out), "--set", "shocks.z_wiki_min=2.5"]) == 0
    validate(pd.read_parquet(out), "events")
