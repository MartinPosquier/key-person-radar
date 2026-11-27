"""Owner: quant session. Tests of kpr.validation.event_study and its notebook (task Q7)."""

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest

from kpr.validation import event_study as ES

matplotlib.use("Agg")
FIXTURES = ES.REPO_ROOT / "tests" / "fixtures"
NOTEBOOK = ES.REPO_ROOT / "notebooks" / "02_event_study.ipynb"


@pytest.fixture(scope="module")
def fx():
    return {n: pd.read_parquet(FIXTURES / f"{n}.parquet") for n in ("events", "prices", "market")}


def test_mean_zero_tests_by_hand():
    out = ES.mean_zero_tests(pd.Series([0.01, 0.02, -0.01, 0.03]))
    assert out["n"] == 4 and out["mean"] == pytest.approx(0.0125)
    sd = np.std([0.01, 0.02, -0.01, 0.03], ddof=1)
    assert out["t"] == pytest.approx(0.0125 / (sd / 2))
    assert out["p_sign"] == pytest.approx(0.625)  # 3 positive out of 4, two-sided
    assert out["share_pos"] == 0.75


def test_mean_zero_tests_degenerate():
    assert np.isnan(ES.mean_zero_tests(pd.Series([], dtype=float))["mean"])
    assert np.isnan(ES.mean_zero_tests(pd.Series([0.01]))["t"])


def test_select_keeps_train_and_validation(fx):
    ev = fx["events"]
    sel = ES.select(ev)
    assert set(sel["split"]) <= {"train", "validation"}
    assert not sel["excluded"].any()
    assert len(ES.select(ev, ("train", "validation", "test"))) > len(sel)


def test_planted_effects_show_in_the_table(fx):
    table = ES.mean_car_table(ES.select(fx["events"]))
    assert list(table.index) == ["negative", "positive"]
    assert table.loc["negative", "mean"] < 0 and table.loc["negative", "p_t"] < 0.05
    assert table.loc["positive", "mean"] > 0


def test_compare_groups():
    ev = pd.DataFrame({"company_id": ["A"] * 20 + ["B"] * 20,
                       "car_0_2": np.r_[np.full(10, 0.05), np.full(10, -0.05),
                                        np.full(10, 0.005), np.full(10, -0.005)]})
    out = ES.compare_groups(ev, {"A": "media", "B": "control"})
    assert out["n_media"] == 20 and out["mean_abs_media"] == pytest.approx(0.05)
    assert out["p_mannwhitney"] < 0.001
    empty = ES.compare_groups(ev, {})
    assert np.isnan(empty["p_welch"])


def test_car_paths_are_consistent_with_the_car_columns(fx):
    from kpr.build import events as E

    t = {n: pd.read_parquet(FIXTURES / f"{n}.parquet")
         for n in ("panel", "earnings", "wiki_views", "gdelt_daily")}
    built = E.build_events(**t, prices=fx["prices"], market=fx["market"], universe={})
    kept = built[~built["excluded"]]
    paths = ES.car_paths(kept, fx["prices"], fx["market"], window=(-10, 10), universe={})
    assert list(paths.columns) == list(range(-10, 11))
    full = paths.dropna()
    assert len(full) >= 10
    car = (full[2] - full[-1]).rename("car")
    merged = kept.set_index("event_id").join(car, how="inner")
    assert np.allclose(merged["car"], merged["car_0_2"])


def test_mean_paths_and_plot(fx):
    ev = ES.select(fx["events"])
    paths = ES.car_paths(ev, fx["prices"], fx["market"], universe={})
    summary = ES.mean_paths(paths, ev.set_index("event_id")["sign"])
    assert set(summary.columns.get_level_values(0)) == {"negative", "positive"}
    neg = summary["negative"]
    assert neg.loc[5, "mean"] < neg.loc[-1, "mean"]  # planted drop after J0
    assert (neg["lo"] <= neg["mean"]).all()
    ax = ES.plot_mean_paths(summary)
    assert len(ax.lines) >= 2


def test_main(monkeypatch, capsys):
    monkeypatch.setenv("KPR_USE_FIXTURES", "1")
    assert ES.main([]) == 0
    assert "CAR[0,2] by sign" in capsys.readouterr().out


def test_notebook_has_no_outputs_and_runs_on_fixtures(monkeypatch):
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(not c["outputs"] and c["execution_count"] is None for c in code)
    source = "".join("".join(c["source"]) for c in code)
    assert 'SPLITS = ("train", "validation")' in source  # the test split stays untouched
    monkeypatch.setenv("KPR_USE_FIXTURES", "1")
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)
    namespace = {"display": lambda *a, **k: None}
    for cell in code:
        exec(compile("".join(cell["source"]), "02_event_study", "exec"), namespace)
    assert "summary" in namespace
