"""Owner: quant session. Tests of kpr.validation.final_numbers (task Q13, docs/final/02)."""

import json
import re

import numpy as np
import pandas as pd
import pytest

from kpr.validation import final_numbers as F
from kpr.validation import natural_experiments as NE

FIXTURES = F.REPO_ROOT / "tests" / "fixtures"
DOC = F.REPO_ROOT / "docs" / "final" / "02_events_and_splits.md"


@pytest.fixture(scope="module")
def fx():
    return {n: pd.read_parquet(FIXTURES / f"{n}.parquet") for n in ("events", "vix")}


def _events(reasons, splits=None, signs=None):
    n = len(reasons)
    return pd.DataFrame({
        "event_id": [f"e{i}" for i in range(n)],
        "company_id": ["A"] * n,
        "date_utc": pd.Timestamp("2018-03-01"), "j0": pd.Timestamp("2018-03-01"),
        "sign": signs or ["negative"] * n,
        "car_0_2": 0.01,
        "excluded": [bool(r) for r in reasons],
        "exclusion_reason": [r or None for r in reasons],
        "split": splits or ["train"] * n,
    })


def test_funnel_counts_each_event_at_its_first_filter():
    ev = _events(["", "earnings", "earnings;extreme_market", "extreme_market",
                  "short_history", "missing_returns", "split_gap", ""])
    out = F.funnel(ev)
    assert out["n_shocks_raw"] == 8
    assert (out["n_removed_earnings"], out["n_after_earnings"]) == (2, 6)
    assert out["n_removed_company"] == 0 and out["n_after_company"] == 6
    assert (out["n_removed_market"], out["n_flagged_market"]) == (1, 2)
    assert out["n_removed_history"] == 1 and out["n_after_history"] == 4
    assert out["n_removed_returns"] == 1 and out["n_removed_gap"] == 1
    assert out["n_events_usable"] == 2
    removed = sum(out[f"n_removed_{k}"] for k, _ in F.FUNNEL)
    assert removed + out["n_events_usable"] == out["n_shocks_raw"]
    assert out["largest_filter"] == "earnings"


def test_funnel_rejects_inconsistent_tables():
    with pytest.raises(ValueError, match="outside the funnel"):
        F.funnel(_events(["typo"]))
    bad = _events(["earnings"]).assign(excluded=False)
    with pytest.raises(ValueError, match="disagree"):
        F.funnel(bad)


def test_groups_signs_and_splits():
    ev = _events(["", "", "", "earnings", ""],
                 splits=["train", "validation", "test", "train", "reserved"],
                 signs=["negative", "positive", "neutral", "negative", "negative"])
    ev["company_id"] = ["A", "B", "A", "A", "A"]
    g = F.group_sign_counts(ev, {"A": "media", "B": "control"})
    assert (g["n_media_neg"], g["n_media_neu"], g["n_media"]) == (1, 1, 2)
    assert (g["n_control_pos"], g["n_control"]) == (1, 1)
    s = F.split_numbers(ev)
    assert (s["n_train"], s["n_train_detected"], s["n_reserved"]) == (1, 2, 1)
    mix = F.sign_mix(ev)
    assert mix["share_neg_raw"] == pytest.approx(0.6) and mix["share_neg_usable"] == 0.5


def test_event_study_leaves_test_out_by_default(fx):
    ev = fx["events"]
    out = F.event_study_numbers(ev, include_test=False)
    kept = ev[~ev.excluded & ev.split.isin(["train", "validation"])]
    assert out["n_neg"] == int((kept.sign == "negative").sum())
    assert out["car_mean_neg"] == pytest.approx(kept.loc[kept.sign == "negative", "car_0_2"].mean())
    with_test = F.event_study_numbers(ev, include_test=True)
    assert with_test["n_neg"] + with_test["n_pos"] + with_test["n_neu"] > (
        out["n_neg"] + out["n_pos"] + out["n_neu"])
    assert "test" in with_test["event_study_splits"]


def test_cv_folds():
    years = [2014, 2015, 2017, 2018, 2019, 2019, 2020, 2021]
    ev = _events([""] * len(years))
    ev["j0"] = pd.to_datetime([f"{y}-06-01" for y in years])
    out = F.cv_fold_numbers(ev, F.E.load_params())
    assert out["cv_fold1_train_years"] == "2015-2017" and out["cv_fold1_val_year"] == 2018
    assert (out["n_cv_fold1_train"], out["n_cv_fold1_val"]) == (3, 1)
    assert (out["n_cv_fold2_val"], out["n_cv_fold4_train"], out["n_cv_fold4_val"]) == (2, 7, 1)


def test_comparability_uses_last_vix_before_j0():
    ev = _events(["", "", ""], splits=["train", "train", "test"],
                 signs=["negative", "positive", "negative"])
    ev["company_id"] = ["A", "B", "A"]
    ev["j0"] = pd.to_datetime(["2018-01-02", "2018-01-04", "2024-07-04"])
    vix = pd.DataFrame({"date": pd.to_datetime(["2018-01-02", "2018-01-03", "2024-07-03"]),
                        "vix": [10.0, 20.0, 30.0]})
    out = F.comparability(ev, {"A": "US", "B": "EU"}, vix)
    assert out["share_neg_train"] == 0.5 and out["share_us_train"] == 0.5
    assert out["vix_train"] == pytest.approx(15.0) and out["vix_test"] == 30.0
    assert out["share_neg_val"] is None and out["vix_val"] is None


def test_m12_run_date(tmp_path):
    assert F.m12_run_date(tmp_path) is None
    (tmp_path / "2026-12-01-other.md").write_text("M7 tuning", encoding="utf-8")
    (tmp_path / "2026-12-20-final-test.md").write_text("Task M12: test run", encoding="utf-8")
    assert F.m12_run_date(tmp_path) == "2026-12-20"


def test_compute_covers_every_placeholder_of_file_02(fx, tmp_path):
    cands = NE.load_candidates()
    cars = pd.DataFrame({"event_id": [cands.event_id.iloc[0]], "car_0_2": [-0.05],
                         "car_0_5": [-0.07]})
    out = F.compute(fx["events"], groups={}, regions={"FAKE1": "US"}, vix=fx["vix"],
                    candidates=cands, candidate_car=cars, decisions_dir=tmp_path)
    json.dumps(out)
    placeholders = set(re.findall(r"\{\{(\w+)\}\}", DOC.read_text(encoding="utf-8")))
    missing = {p for p in placeholders if p not in out and not p.startswith("param_")}
    assert not missing
    assert {p for p in placeholders if p.startswith("param_")} <= set(out)
    assert out["n_natural_candidates"] == len(cands) and out["n_natural_with_car"] == 1
    assert out["natural_experiments"][0]["car_0_2"] == -0.05
    assert not any(isinstance(v, float) and np.isnan(v) for v in out.values())


def test_main(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("KPR_USE_FIXTURES", "1")
    assert F.main([]) == 0
    assert json.loads(capsys.readouterr().out)["_meta"]["fixtures"] is True
    target = tmp_path / "quant.json"
    assert F.main(["--out", str(target)]) == 0
    assert json.loads(target.read_text(encoding="utf-8"))["n_shocks_raw"] == 40
