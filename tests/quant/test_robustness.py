"""Owner: quant session. Tests of kpr.validation.robustness (task Q9, dossier §11.4)."""

import numpy as np
import pandas as pd
import pytest

from kpr.validation import robustness as RB

FIXTURES = RB.E.REPO_ROOT / "tests" / "fixtures"
PLANTED = {"p_alpha": 3.0, "p_beta": 0.8, "p_gamma": 1.5, "p_delta": 0.3, "p_eps": 1.0}


@pytest.fixture(scope="module")
def tables():
    return {n: pd.read_parquet(FIXTURES / f"{n}.parquet") for n in RB.TABLE_NAMES}


@pytest.fixture(scope="module")
def reference(tables):
    return RB.run_variant(tables, {}, universe={})


def test_default_variants_follow_the_dossier():
    v = RB.default_variants()
    assert v["reference"] == {}
    assert v["z_wiki_2.5"] == {"shocks": {"z_wiki_min": 2.5}}
    assert v["z_wiki_4"] == {"shocks": {"z_wiki_min": 4.0}}
    assert {"window_0_1", "window_0_5", "window_m1_1", "market_adjusted", "ff3",
            "earnings_2", "earnings_none"} <= set(v)
    assert "z_wiki_3" not in v and "earnings_1" not in v  # the reference values


def test_sensitivity_score_recovers_planted_ordering():
    ev = pd.read_parquet(FIXTURES / "events.parquet")
    score = RB.sensitivity_score(ev.assign(split="train"), min_events=4)
    planted = pd.Series(PLANTED).reindex(score.index)
    assert len(score) >= 4
    assert score.corr(planted, method="spearman") >= 0.8


def test_mean_abs_car_score():
    ev = pd.DataFrame({"person_id": ["a"] * 3 + ["b"] * 2, "car_0_2": [0.1, -0.1, 0.1, 1, 1],
                       "excluded": False, "split": "train"})
    s = RB.mean_abs_car_score(ev)
    assert s.to_dict() == {"a": pytest.approx(0.1)}  # b has too few events


def test_window_variant_moves_the_target(tables, reference):
    v = RB.run_variant(tables, {"returns": {"car_window": [0, 1]}}, universe={})
    both = v.set_index("event_id")["car_0_2"].to_frame("v").join(
        reference.set_index("event_id")["car_0_1"]).dropna()
    assert len(both) > 10 and np.allclose(both["v"], both["car_0_1"])


def test_market_adjusted_has_no_history_requirement(tables):
    v = RB.run_variant(tables, {"returns": {"model": "market_adjusted"}}, universe={})
    assert not v["exclusion_reason"].fillna("").str.contains("short_history").any()


def test_earnings_none_disables_the_filter(tables):
    v = RB.run_variant(tables, {"filters": {"earnings_window_sessions": RB.NO_FILTER}},
                       universe={})
    assert not v["exclusion_reason"].fillna("").str.contains("earnings").any()


def test_ff3_variant_runs_with_factors(tables):
    m = tables["market"]
    gspc = m[m.index_id == "^GSPC"].set_index("date")["ret"]
    rng = np.random.default_rng(0)
    factors = pd.DataFrame({"mkt_rf": gspc, "smb": rng.normal(0, 0.005, len(gspc)),
                            "hml": rng.normal(0, 0.005, len(gspc)), "rf": 0.0}, index=gspc.index)
    v = RB.run_variant(tables, {"returns": {"model": "ff3"}}, universe={},
                       factors_for=lambda _c: factors)
    assert v["car_0_2"].notna().sum() > 10


def test_run_all_table(tables):
    variants = {k: v for k, v in RB.default_variants().items()
                if k in ("reference", "z_wiki_4", "ff3")}
    table, matrix = RB.run_all(tables, variants=variants, universe={},
                               score_fn=lambda ev: RB.sensitivity_score(ev, min_events=3))
    assert list(table.index) == ["reference", "z_wiki_4", "ff3"]
    assert table.loc["reference", "spearman"] == pytest.approx(1.0)
    assert table.loc["ff3", "note"].startswith("skipped")
    assert "ff3" not in matrix.columns
    assert table.loc["z_wiki_4", "kept"] <= table.loc["reference", "kept"]
    with pytest.raises(ValueError, match="reference"):
        RB.run_all(tables, variants={"z_wiki_4": {}})


def test_main(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("KPR_USE_FIXTURES", "1")
    assert RB.main(["--only", "window_0_1", "--out-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "window_0_1" in out and "z_wiki_4" not in out
    assert (tmp_path / "robustness.csv").exists()
    assert (tmp_path / "robustness_scores.csv").exists()
