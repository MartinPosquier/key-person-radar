"""Owner: ml session. Tests for kpr.build.tone (task M5, dossier §8.3).

The real FinBERT model is exercised only when KPR_RUN_FINBERT=1 (it needs torch,
transformers and a ~440 MB download); every other test injects a fake scorer.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from kpr.build import tone as T
from kpr.schemas import validate


class FakeScorer:
    """'Trouble' -> negative, 'Praise' -> positive, else neutral. Records batch sizes."""

    def __init__(self, fail_after: int | None = None):
        self.batches: list[int] = []
        self.fail_after = fail_after

    def __call__(self, titles):
        if self.fail_after is not None and len(self.batches) >= self.fail_after:
            raise RuntimeError("simulated crash")
        self.batches.append(len(titles))
        rows = [[0.1, 0.8, 0.1] if "Trouble" in t else [0.7, 0.1, 0.2] if "Praise" in t
                else [0.1, 0.1, 0.8] for t in titles]
        return np.array(rows)


@pytest.fixture
def headlines(monkeypatch):
    monkeypatch.setenv("KPR_USE_FIXTURES", "1")
    from kpr.utils.io import read_table

    return read_table("headlines")


def test_builds_a_valid_headlines_tone_table(headlines, tmp_path):
    scorer = FakeScorer()
    tone = T.build_tone(headlines, scorer, tmp_path / "c.parquet", "fake", batch_size=32)
    validate(tone, "headlines_tone")
    assert len(tone) == headlines.groupby(["person_id", "date"]).ngroups
    assert (tone["n_headlines"] == 3).all()
    assert max(scorer.batches) <= 32 and sum(scorer.batches) == len(headlines)


def test_daily_aggregation_by_hand():
    scored = pd.DataFrame({
        "person_id": ["a", "a", "b"], "date": pd.to_datetime(["2021-01-04"] * 3),
        "headline_id": ["h1", "h2", "h3"],
        "p_positive": [0.6, 0.2, 0.1], "p_negative": [0.2, 0.6, 0.8], "p_neutral": 0.2,
    })
    out = T.aggregate_daily(scored).set_index("person_id")
    assert out.loc["a", "finbert_pos"] == pytest.approx(0.4)
    assert out.loc["a", "finbert_net"] == pytest.approx(0.0)
    assert out.loc["b", "finbert_net"] == pytest.approx(-0.7)
    assert out.loc["a", "n_headlines"] == 2


def test_cache_avoids_rescoring(headlines, tmp_path):
    cache = tmp_path / "c.parquet"
    first = T.build_tone(headlines, FakeScorer(), cache, "fake")
    again = FakeScorer()
    second = T.build_tone(headlines, again, cache, "fake")
    assert again.batches == []
    pd.testing.assert_frame_equal(first, second)
    other_model = FakeScorer()
    T.build_tone(headlines, other_model, cache, "another-model")
    assert sum(other_model.batches) == len(headlines)


def test_interrupted_run_resumes(headlines, tmp_path, monkeypatch):
    monkeypatch.setattr(T, "SAVE_EVERY_BATCHES", 1)
    cache = tmp_path / "c.parquet"
    with pytest.raises(RuntimeError):
        T.build_tone(headlines, FakeScorer(fail_after=2), cache, "fake", batch_size=10)
    assert len(T.load_cache(cache)) == 20  # two batches saved before the crash
    resumed = FakeScorer()
    T.build_tone(headlines, resumed, cache, "fake", batch_size=10)
    assert sum(resumed.batches) == len(headlines) - 20


def test_alias_filter():
    h = pd.DataFrame({
        "person_id": ["jh", "jh", "jh", "x"],
        "title": ["Jensen Huang unveils chip", "Nvidia shares rise", "Huangshan travel",
                  "anything"],
    })
    kept = T.filter_by_alias(h, {"jh": ["jensen huang", "huang"]})
    assert kept["title"].tolist() == ["Jensen Huang unveils chip", "anything"]


def test_signs_follow_the_fake_scorer(headlines, tmp_path):
    tone = T.build_tone(headlines, FakeScorer(), tmp_path / "c.parquet", "fake")
    trouble = headlines[headlines["title"].str.startswith("Trouble")]
    neg_days = tone.merge(trouble[["person_id", "date"]].drop_duplicates())
    assert (neg_days["finbert_net"] < 0).all()


@pytest.mark.skipif(os.environ.get("KPR_RUN_FINBERT") != "1",
                    reason="set KPR_RUN_FINBERT=1 to run the real model")
def test_real_finbert_orders_obvious_headlines():
    pytest.importorskip("transformers")
    probs = T.FinBertScorer()(["Company profits surge to record high",
                               "CEO arrested for fraud, shares collapse"])
    assert probs.shape == (2, 3)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)
    assert probs[0, 0] > probs[0, 1] and probs[1, 1] > probs[1, 0]
