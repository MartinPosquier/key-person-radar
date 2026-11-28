Date: 2026-11-27
Session: data
Decision: headlines of shock days (task D12)
- Target days: UTC shock day ± 1 day for every event (dossier §8.3 step 6).
- Until 2023: FNSPID `All_external.csv` (5.7 GB: titles, tickers, URLs, publishers; the 23 GB
  full-text file is not needed). Subset kept: our US tickers or a person alias in the title.
- 2024-2025: titles rebuilt from GDELT URL slugs (longest path segment, ids and extensions
  removed, at least 4 words); one BigQuery query per year restricted to the partitions of the
  target days, dry run and 30 GB cap.
- Only titles naming the person (alias, word boundaries) are kept. headline_id = hash of the
  URL; source = "fnspid:<publisher>" or "gdelt_url".
- Runs on the events fixture until Q5 is merged; the real run needs the events table, the
  FNSPID subset and, for 2024-2025, BigQuery access.
Why: dossier §8.3.
