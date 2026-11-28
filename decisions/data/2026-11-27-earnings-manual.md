Date: 2026-11-27
Session: data
Decision: manual earnings dates (task D9)
- File config/earnings_manual.csv (company_id, date, quarter, source, checked_by, note) for
  the 26 companies without 8-K filings: 23 European listings plus Shopify, Spotify,
  BioNTech.
- `--template` keeps every existing row, pre-fills the release dates yfinance knows (source
  `yfinance`, local date, to confirm: 735 dates) and adds one empty row per company-year
  without any date (116), so that gaps are visible.
- The `earnings` table = EDGAR dates + every dated manual row; kpr.collect.edgar writes its
  dates to raw/edgar/earnings_edgar.parquet and calls the merge, so both orders work.
- Completeness check: 3-6 releases per full year in the US, 2-5 in Europe; each company's
  first year is not checked (partial).
Why: dossier §5.6, §6.4 ("compléter les trous à la main").
