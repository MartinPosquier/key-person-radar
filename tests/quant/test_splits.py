"""Owner: quant session. Tests of kpr.build.splits (task Q6, dossier §9.5)."""

import pandas as pd
import pytest

from kpr.build import events as E
from kpr.build import splits as S

PARAMS = E.load_params()
SPLITS = PARAMS["splits"]


def _events(dates, reserved=None, reasons=None):
    n = len(dates)
    j0 = pd.to_datetime(pd.Series(dates))
    return pd.DataFrame({
        "event_id": [f"e{i}" for i in range(n)], "date_utc": j0, "j0": j0,
        "sign": "negative", "is_natural_experiment": reserved or [False] * n,
        "excluded": [r is not None for r in (reasons or [None] * n)],
        "exclusion_reason": pd.Series(reasons or [None] * n, dtype=object),
    })


def test_gap_is_the_10_business_days_before_each_boundary():
    # Validation starts on Sat 2022-01-01: the gap is Mon 2021-12-20 .. Fri 2021-12-31.
    dates = pd.Series(pd.to_datetime(["2021-12-17", "2021-12-20", "2021-12-31", "2022-01-03",
                                      "2023-12-15", "2023-12-18", "2023-12-29", "2024-01-02"]))
    gap = S.boundary_gap(dates, SPLITS, 10)
    assert list(gap) == [False, True, True, False, False, True, True, False]


def test_gap_matches_the_ml_cv_convention():
    # kpr.models.cv.ExpandingYearSplit drops training events with J0 >= start - BDay(gap).
    start = pd.Timestamp("2022-01-01")
    cutoff = start - pd.offsets.BDay(10)
    days = pd.Series(pd.bdate_range("2021-11-01", "2021-12-31"))
    assert list(S.boundary_gap(days, SPLITS, 10)) == list(days >= cutoff)


def test_zero_gap():
    assert not S.boundary_gap(pd.Series(pd.to_datetime(["2021-12-31"])), SPLITS, 0).any()


def test_assign_splits_flags_gap_events_and_keeps_other_reasons():
    ev = _events(["2019-05-06", "2021-12-28", "2021-12-29", "2021-12-30", "2022-06-01"],
                 reserved=[False, False, False, True, False],
                 reasons=[None, None, "earnings", None, None])
    out = S.assign_splits(ev, SPLITS, 10).set_index("event_id")
    assert list(out["split"]) == ["train", "train", "train", "reserved", "validation"]
    assert out.loc["e1", "exclusion_reason"] == "split_gap" and out.loc["e1", "excluded"]
    assert out.loc["e2", "exclusion_reason"] == "earnings;split_gap"
    # Reserved events are already out of training: no gap flag on them.
    assert out.loc["e3", "exclusion_reason"] is None and not out.loc["e3", "excluded"]
    assert not out.loc["e0", "excluded"] and not out.loc["e4", "excluded"]
    assert ev["exclusion_reason"].isna().sum() == 4  # input untouched


def test_summary_counts():
    ev = S.assign_splits(_events(["2019-05-06", "2021-12-28", "2022-06-01"]), SPLITS, 10)
    table = S.summary(ev)
    assert table.loc["train", "events"] == 2 and table.loc["train", "kept"] == 1
    assert table.loc["train", "split_gap"] == 1
    assert table.loc["validation", "negative"] == 1


def test_finalize_applies_the_gap():
    ev = _events(["2021-12-28"]).assign(
        person_id="p", company_id="C", z_wiki=4.0, z_news=3.0, n_articles=10, tone_rel=-2.0,
        tone_neg_share=0.5, car_0_2=0.01, car_m5_m1=0.0, car_0_1=0.01, car_0_5=0.01)
    cand = pd.DataFrame({"event_id": [], "company_id": [], "date": pd.to_datetime([])})
    out = E.finalize(ev, cand)
    assert out.loc[0, "excluded"] and out.loc[0, "exclusion_reason"] == "split_gap"


def test_main_prints_the_fixture_split(monkeypatch, capsys):
    monkeypatch.setenv("KPR_USE_FIXTURES", "1")
    assert S.main([]) == 0
    out = capsys.readouterr().out
    assert "reserved" in out and "kept" in out


@pytest.mark.parametrize("bad", [{"validation": ["2021-06-01", "2023-12-31"]},
                                 {"test": ["2023-06-01", "2025-12-31"]}])
def test_overlapping_periods_are_rejected(bad):
    with pytest.raises(ValueError, match="overlap"):
        S.boundary_gap(pd.Series(pd.to_datetime(["2020-01-02"])), dict(SPLITS, **bad), 10)
