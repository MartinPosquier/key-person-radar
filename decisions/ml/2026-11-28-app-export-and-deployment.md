Date: 2026-11-28
Session: ml
Decision: `python -m kpr.scores.export_app` writes the app files of app/data/ from the shared
tables: the `scores` table as is; `events` plus `car_pred` (the M7 best model's prediction
for every event, display only); daily views, adjusted close and market-adjusted return
(stock minus its index, beta = 1) for tracked persons; entities with the display name taken
from the English Wikipedia title in persons.yaml and the company name, region, sector and
group from universe.yaml; meta.json with the §10.7 sanity checks; case_study.json converted
from quant's processed/validation/case_nvidia.json when it exists (the demo case is deleted
otherwise). A fixture run must name an output folder, so the committed app data is never
overwritten by synthetic data by accident. The app gets its own app/requirements.txt
(streamlit, plotly, pandas, pyarrow) so that Streamlit Community Cloud does not install
torch and the ML stack; deployment steps are in app/README.md.
Why: dossier §12.3 (precomputed files only, st.cache_data, free Streamlit Community Cloud
from a public repo, no API key), §12.4 (honest display), §10.7 (sanity checks before
publishing, documented whatever the result).
Alternatives considered: daily abnormal returns from the market model (needs the quant
session's per-day estimates, not in any contract; the daily series is illustrative, the
event CARs shown come from the events table); a display-name column in persons.yaml (owned
by the data session; `name:` is read if they add it).
