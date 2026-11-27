"""Owner: quant session. Tests of kpr.validation.hypotheses (task Q11, dossier §11.5)."""

import numpy as np
import pandas as pd
import pytest

from kpr.validation import hypotheses as H

FIXTURES = H.ES.REPO_ROOT / "tests" / "fixtures"


@pytest.fixture(scope="module")
def fx():
    return {n: pd.read_parquet(FIXTURES / f"{n}.parquet") for n in ("scores", "mandates")}


def _scores(rows):
    cols = ["person_id", "company_id", "year", "s_minus", "s_plus", "v_pct"]
    df = pd.DataFrame(rows, columns=cols)
    df["s_minus_raw"], df["s_plus_raw"] = df["s_minus"], df["s_plus"]
    for col, (lo, hi) in H.INTERVALS.items():
        df[lo], df[hi] = df[col] - 1, df[col] + 1
    return df


def test_h2_on_fixtures_finds_the_planted_asymmetry(fx):
    out = H.h2_asymmetry(fx["scores"])
    assert out["n"] == 5 and out["median_ratio"] == pytest.approx(2.0)
    assert out["share_above_1"] == 1.0 and out["p_sign"] == pytest.approx(1 / 32)


def test_h2_counts_each_person_once():
    s = _scores([["a", "X", y, 2.0, 1.0, 1.0] for y in range(2015, 2025)]
                + [["b", "Y", 2020, 1.0, 2.0, 1.0]])
    out = H.h2_asymmetry(s)
    assert out["n"] == 2 and out["share_above_1"] == 0.5


def test_h3_founders_and_media():
    rng = np.random.default_rng(0)
    rows, mandates, groups = [], [], {}
    for i in range(20):
        founder = i < 10
        cid = f"C{i}"
        groups[cid] = "media" if i % 2 == 0 else "control"
        level = 6.0 if founder else 2.0
        rows.append([f"p{i}", cid, 2020, level + rng.normal(0, 0.5), 1.0, level])
        mandates.append({"person_id": f"p{i}", "company_id": cid, "role": "CEO",
                         "start": pd.Timestamp("2010-01-01"), "end": pd.NaT,
                         "founder": founder})
    out = H.h3_groups(_scores(rows), pd.DataFrame(mandates), groups)
    f = out.loc[("founders_vs_others", "s_minus")]
    assert f["n_1"] == 10 and f["mean_1"] > f["mean_2"] and f["p_mannwhitney"] < 0.001
    m = out.loc[("media_vs_control", "s_minus")]
    assert m["n_1"] == 10 and m["n_2"] == 10 and m["p_mannwhitney"] > 0.05


def test_successions_from_mandates(fx):
    succ = H.successions(fx["mandates"])
    assert succ.to_dict("records") == [{"company_id": "FAKE3", "year": 2021,
                                        "outgoing": "p_delta", "incoming": "p_eps"}]


def test_h4_on_fixtures(fx):
    out = H.h4_stability(fx["scores"], fx["mandates"])
    v = out.loc["v_pct"]
    assert v["n_pairs"] == 5 and v["mean_abs_change"] == 0.0
    assert v["n_successions"] == 1 and v["mean_abs_jump"] == pytest.approx(2.0 - 0.6)


def test_h4_spearman_and_jump_test():
    rows = []
    for i in range(12):
        for y in (2019, 2020, 2021):
            rows.append([f"p{i}", f"C{i}", y, i + 0.1 * (y - 2019), 1.0, float(i)])
    s = _scores(rows)
    out = H.h4_stability(s, pd.DataFrame(columns=["person_id", "company_id", "role", "start",
                                                  "end", "founder"]))
    assert out.loc["s_minus", "spearman_yoy"] == pytest.approx(1.0)
    assert out.loc["s_minus", "n_successions"] == 0


def test_h5_overlap_logic():
    s = _scores([["jensen_huang", "NVDA", 2025, 8.0, 7.0, 5.0],
                 ["patrick_pouyanne", "TTE", 2025, 3.0, 6.5, 4.5],
                 ["patrick_pouyanne", "TTE", 2024, 3.0, 3.0, 3.0]])
    out = H.h5_compare(s)
    assert out.loc["s_minus", "verdict"] == "jensen_huang higher (intervals disjoint)"
    assert out.loc["s_plus", "overlap"] and out.loc["v_pct", "overlap"]
    assert out.loc["s_minus", "year"] == 2025
    with pytest.raises(ValueError, match="no common"):
        H.h5_compare(s, year=None, persons=("jensen_huang", "nobody"))


def test_main(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("KPR_USE_FIXTURES", "1")
    assert H.main(["--h5", "p_alpha", "p_gamma", "--out-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "== H2" in out and "== H5" in out
    assert {p.name for p in tmp_path.iterdir()} == {
        "hypotheses_h2.csv", "hypotheses_h3.csv", "hypotheses_h4.csv", "hypotheses_h5.csv"}
    assert H.main([]) == 0  # default H5 persons are not in the fixtures: skipped, no crash
