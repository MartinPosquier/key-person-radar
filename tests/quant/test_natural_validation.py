"""Owner: quant session. Tests of the natural-experiment validation (task Q10, §11.2)."""

import numpy as np
import pandas as pd
import pytest

from kpr.validation import natural_experiments as ne

FIXTURES = ne.REPO_ROOT / "tests" / "fixtures"


@pytest.fixture(scope="module")
def fx():
    return {n: pd.read_parquet(FIXTURES / f"{n}.parquet")
            for n in ("events", "prices", "market", "scores", "wiki_views")}


@pytest.fixture(scope="module")
def candidates(fx):
    """Planted 2021 shocks of the fixtures, used as if they were natural experiments."""
    ev = fx["events"][fx["events"].date_utc.dt.year == 2021]
    return pd.DataFrame({"event_id": "NE_" + ev.event_id, "person_name": ev.person_id,
                         "person_id": ev.person_id, "company_id": ev.company_id,
                         "date": ev.date_utc}).reset_index(drop=True)


def test_slug():
    assert ne.slug("Lars Fruergaard Jørgensen") == "lars_fruergaard_jorgensen"
    assert ne.slug("Steve Easterbrook") == "steve_easterbrook"
    assert ne.slug("Uğur Şahin") == "ugur_sahin"


def test_attach_person_ids_uses_persons_yaml_aliases():
    known = {"jensen_huang": "jensen_huang", "jen_hsun_huang": "jensen_huang"}
    c = pd.DataFrame({"person_name": ["Jen-Hsun Huang", "Bob Iger"]})
    out = ne.attach_person_ids(c, known)
    assert list(out["person_id"]) == ["jensen_huang", "bob_iger"]
    kept = ne.attach_person_ids(c.assign(person_id=["x", "y"]), known)
    assert list(kept["person_id"]) == ["x", "y"]


def test_load_person_ids(tmp_path):
    path = tmp_path / "persons.yaml"
    path.write_text("persons:\n- person_id: jensen_huang\n  aliases: [jensen huang, "
                    "jen-hsun huang]\n", encoding="utf-8")
    assert ne.load_person_ids(path)["jen_hsun_huang"] == "jensen_huang"


def test_ex_ante_scores_use_the_previous_year():
    scores = pd.DataFrame({"person_id": ["a", "a", "b"], "company_id": ["X", "X", "Y"],
                           "year": [2020, 2021, 2020], "v_pct": [1.0, 9.0, 3.0],
                           "s_minus": [1.0, 9.0, 3.0], "s_plus": [1.0, 9.0, 3.0]})
    c = pd.DataFrame({"event_id": ["e1", "e2", "e3"], "person_id": ["a", "b", "c"],
                      "company_id": ["X", "Z", "X"],
                      "date": pd.to_datetime(["2021-05-01", "2021-05-01", "2021-05-01"])})
    out = ne.ex_ante_scores(c, scores).set_index("event_id")
    assert out.loc["e1", "v_pct"] == 1.0 and out.loc["e1", "score_source"] == "same_company"
    assert out.loc["e2", "v_pct"] == 3.0 and out.loc["e2", "score_source"] == "other_company"
    assert np.isnan(out.loc["e3", "v_pct"]) and out.loc["e3", "score_source"] == "none"


def test_naive_score_is_the_mean_of_the_previous_year():
    days = pd.date_range("2020-01-01", "2021-12-31")
    views = pd.DataFrame({"page_id": "a", "date": days,
                          "views_total": np.where(days.year == 2020, 10.0, 1000.0)})
    c = pd.DataFrame({"event_id": ["e"], "person_id": ["a"],
                      "date": pd.to_datetime(["2021-01-01"])})
    assert ne.naive_score(c, views)["e"] == pytest.approx(10.0)


def test_bootstrap_spearman():
    x = pd.Series(np.arange(30.0))
    rho, lo, hi = ne.bootstrap_spearman(x, x * 2 + 1)
    assert rho == pytest.approx(1.0) and lo == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    rho, lo, hi = ne.bootstrap_spearman(pd.Series(rng.normal(size=200)),
                                        pd.Series(rng.normal(size=200)))
    assert lo < 0 < hi
    assert np.isnan(ne.bootstrap_spearman(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))[0])


def test_candidate_cars_match_the_returns_module(fx, candidates):
    out = ne.candidate_cars(candidates, fx["prices"], fx["market"], universe={})
    assert len(out) == len(candidates)
    assert (out["j0"] == out["date"]).all()  # fixtures: every weekday is a session
    ev = fx["events"].set_index("event_id")
    both = out.set_index(out["event_id"].str[3:])[["car_0_2"]].join(ev["car_0_2"], rsuffix="_t")
    assert both["car_0_2"].corr(both["car_0_2_t"]) > 0.5


def test_unknown_company_is_skipped(fx, candidates):
    c = candidates.head(2).assign(company_id=["NOPE", candidates.company_id.iloc[1]])
    out = ne.candidate_cars(c, fx["prices"], fx["market"], universe={})
    assert list(out["company_id"]) == [candidates.company_id.iloc[1]]


def test_validation_on_fixtures_beats_the_naive_score(fx, candidates):
    table, summary = ne.validate_scores(candidates, fx["prices"], fx["market"], fx["scores"],
                                        fx["wiki_views"], universe={}, person_ids={})
    assert (table["score_source"] == "same_company").all()
    assert list(summary.index) == ["v_pct", "s_minus", "s_plus", "naive_views"]
    v = summary.loc["v_pct"]
    assert v["n"] >= 20 and v["spearman"] > 0 and v["lo"] > 0
    assert v["vs_naive"] > 0


def test_main_on_fixtures(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("KPR_USE_FIXTURES", "1")
    assert ne.main(["--out-dir", str(tmp_path)]) == 0  # real candidates, fake prices: empty
    assert (tmp_path / "natural_experiments_summary.csv").exists()
    assert "naive_views" in capsys.readouterr().out
