From: quant
To: data
Need: for the Nvidia 2026 case study (task Q12), three parquet files kept OUT of the shared study tables (for example under interim/case_nvidia/):
- `views_2026.parquet`: wiki_views columns, page_id = jensen_huang, 2025-01-01 to today (2025 is needed for the 60-day z-score reference);
- `prices_2025_2026.parquet`: prices columns for NVDA, AMD, INTC, ASML, 2024-06-01 to today (the estimation window needs 250 sessions before the event);
- `market_2025_2026.parquet`: market columns for ^GSPC and the STOXX 600 index, same dates.
Why: dossier §11.6. The study tables stop at 2025-12-31 (events.py refuses later dates), so the case data must stay separate. Command: `python -m kpr.validation.case_nvidia --views ... --prices ... --market ...`.
Urgency: low (week S4)
