Date: 2026-11-27
Session: ml
Decision: FinBERT (ProsusAI/finbert) scores each headline title, truncated at 64 tokens, in
batches of 32; probabilities are cached per (model name, headline_id) in
processed/tone/finbert_cache.parquet and saved every 20 batches, so runs are resumable.
Daily aggregation: finbert_pos and finbert_neg are the mean probabilities, finbert_net the
mean of P(positive) - P(negative), n_headlines the number of distinct headlines. Headlines
are kept only if they contain one of the person's aliases from config/persons.yaml (whole
word, case-insensitive); persons without aliases keep all their headlines.
Why: dossier §8.3 (titles only, batches of 32, three probabilities, mean probabilities and
net score, alias filter, run only on shock days and neighbours, which the headlines table
of task D12 already restricts to).
Alternatives considered: caching by title text (rejected: the same title can belong to two
persons, and headline_id is the frozen key of the headlines table); scoring full articles
(out of scope, §8.3).
