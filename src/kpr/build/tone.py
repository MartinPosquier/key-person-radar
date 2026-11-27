"""FinBERT inference on headlines, aggregated per person and day.

Owner: ml session. Task M5 in PARALLEL_SESSIONS.md §7. Dossier §8.3.

Steps (§8.3): keep the headlines that name the person (or an alias), score them with
FinBERT in batches of 32 (3 probabilities per headline), then aggregate per (person, day):
mean P(positive), mean P(negative), net score = P(positive) - P(negative), and the number of
headlines. The result is the `headlines_tone` table.

Scores are cached per (model, headline_id) in processed/tone/, and the cache is saved after
every chunk of batches, so an interrupted run resumes where it stopped and a re-run only
scores new headlines. The `headlines` table already covers shock days and their neighbours
only (task D12), which keeps the run short.

torch and transformers (requirements/nlp.txt) are imported only when the real model runs;
tests inject a fake scorer.

Run: python -m kpr.build.tone [--batch-size 32] [--limit N]
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kpr.utils.io import data_dir, read_table, use_fixtures, write_table
from kpr.utils.logging import get_logger

log = get_logger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
LABELS = ("positive", "negative", "neutral")
CACHE_COLUMNS = ["model", "headline_id", "p_positive", "p_negative", "p_neutral"]
SAVE_EVERY_BATCHES = 20

Scorer = Callable[[list[str]], np.ndarray]  # titles -> (n, 3) probabilities in LABELS order


def load_params() -> dict:
    ml = yaml.safe_load((CONFIG_DIR / "params" / "ml.yaml").read_text(encoding="utf-8"))
    return dict(ml["finbert"])


def load_aliases() -> dict[str, list[str]]:
    """person_id -> lowercase names to look for in titles (config/persons.yaml)."""
    raw = yaml.safe_load((CONFIG_DIR / "persons.yaml").read_text(encoding="utf-8")) or {}
    aliases: dict[str, list[str]] = {}
    for entry in raw.get("persons") or []:
        names = [a.lower() for a in entry.get("aliases") or []]
        aliases.setdefault(entry["person_id"], [])
        aliases[entry["person_id"]] += [n for n in names if n not in aliases[entry["person_id"]]]
    return aliases


class FinBertScorer:
    """Lazy FinBERT wrapper: probabilities (positive, negative, neutral) for a list of titles."""

    def __init__(self, model_name: str = "ProsusAI/finbert", max_length: int = 64):
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer = self._model = self._order = None

    def _load(self) -> None:
        import torch  # noqa: F401  (requirements/nlp.txt)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()
        id2label = {i: lab.lower() for i, lab in self._model.config.id2label.items()}
        self._order = [next(i for i, lab in id2label.items() if lab == name) for name in LABELS]

    def __call__(self, titles: list[str]) -> np.ndarray:
        if self._model is None:
            self._load()
        import torch

        enc = self._tokenizer(titles, padding=True, truncation=True,
                              max_length=self.max_length, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(self._model(**enc).logits, dim=-1).numpy()
        return probs[:, self._order]


def filter_by_alias(headlines: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    """Keep titles naming their person (§8.3). Persons without aliases keep all their titles."""
    def keep(row) -> bool:
        names = aliases.get(row.person_id)
        if not names:
            return True
        title = row.title.lower()
        return any(re.search(rf"\b{re.escape(n)}\b", title) for n in names)

    if headlines.empty:
        return headlines
    return headlines[[keep(r) for r in headlines.itertuples(index=False)]]


def _save_cache(cache: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        cache.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_cache(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=CACHE_COLUMNS)


def score_headlines(headlines: pd.DataFrame, scorer: Scorer, cache_path: Path,
                    model_name: str, batch_size: int = 32) -> pd.DataFrame:
    """Per-headline probabilities, scoring only what the cache does not hold yet."""
    cache = load_cache(cache_path)
    done = set(cache.loc[cache["model"] == model_name, "headline_id"])
    todo = headlines.drop_duplicates("headline_id")
    todo = todo[~todo["headline_id"].isin(done)]
    log.info("FinBERT: %d headlines cached, %d to score", len(done), len(todo))

    new_parts: list[pd.DataFrame] = []
    for k, start in enumerate(range(0, len(todo), batch_size), start=1):
        batch = todo.iloc[start:start + batch_size]
        probs = np.asarray(scorer(batch["title"].tolist()), dtype=float)
        new_parts.append(pd.DataFrame({
            "model": model_name, "headline_id": batch["headline_id"].to_numpy(),
            "p_positive": probs[:, 0], "p_negative": probs[:, 1], "p_neutral": probs[:, 2],
        }))
        if k % SAVE_EVERY_BATCHES == 0 or start + batch_size >= len(todo):
            cache = pd.concat([c for c in (cache, *new_parts) if len(c)], ignore_index=True)
            new_parts = []
            _save_cache(cache, cache_path)
            log.info("FinBERT: %d / %d scored", min(start + batch_size, len(todo)), len(todo))

    mine = cache[cache["model"] == model_name].drop(columns="model")
    return headlines.merge(mine, on="headline_id", how="inner")


def aggregate_daily(scored: pd.DataFrame) -> pd.DataFrame:
    """The `headlines_tone` table: one row per (person, day)."""
    cols = ["person_id", "date", "finbert_pos", "finbert_neg", "finbert_net", "n_headlines"]
    if scored.empty:
        return pd.DataFrame(columns=cols)
    out = (
        scored.assign(net=scored["p_positive"] - scored["p_negative"])
        .groupby(["person_id", "date"], as_index=False)
        .agg(finbert_pos=("p_positive", "mean"), finbert_neg=("p_negative", "mean"),
             finbert_net=("net", "mean"), n_headlines=("headline_id", "nunique"))
    )
    out["n_headlines"] = out["n_headlines"].astype("int64")
    return out[cols]


def build_tone(headlines: pd.DataFrame, scorer: Scorer, cache_path: Path, model_name: str,
               batch_size: int = 32, aliases: dict[str, list[str]] | None = None
               ) -> pd.DataFrame:
    kept = filter_by_alias(headlines, aliases or {})
    log.info("headlines naming their person: %d / %d", len(kept), len(headlines))
    scored = score_headlines(kept, scorer, cache_path, model_name, batch_size)
    return aggregate_daily(scored)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FinBERT tone of headlines (dossier §8.3)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="score the first N headlines")
    args = parser.parse_args(argv)
    params = load_params()
    batch_size = args.batch_size or params["batch_size"]

    headlines = read_table("headlines")
    if args.limit:
        headlines = headlines.head(args.limit)
    name = "finbert_cache_fixtures.parquet" if use_fixtures() else "finbert_cache.parquet"
    cache_path = data_dir() / "processed" / "tone" / name
    tone = build_tone(headlines, FinBertScorer(params["model"]), cache_path, params["model"],
                      batch_size, load_aliases())
    log.info("headlines_tone: %d person-days, mean net score %.3f", len(tone),
             tone["finbert_net"].mean() if len(tone) else float("nan"))
    if use_fixtures() or args.limit:
        log.info("fixture or partial run: headlines_tone not written")
        return 0
    log.info("written to %s", write_table(tone, "headlines_tone"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
