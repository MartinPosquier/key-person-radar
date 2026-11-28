From: data
To: user
Need: a Google Cloud project for the GDELT dry run (task D5), then the GDELT collection (D8)
Why: GDELT's history is only on BigQuery (dossier §5.3). No project, credentials or gcloud CLI
exist on this machine, and `GCP_PROJECT` is empty, so the dry run could not run. The code is
ready: `python -m kpr.collect.gdelt --dry-run 2019-06` estimates one month, extrapolates to
2015-2025 and applies the D5 rule (under 50 % of the 1 TiB free monthly quota -> GDELT,
otherwise FNSPID). Every real query carries `maximum_bytes_billed`.
How (about 10 minutes, no payment needed):
1. https://console.cloud.google.com/ -> create a project (e.g. `kpr-gdelt`). Use the
   BigQuery **sandbox** (no billing account): it gives the free quota and cannot be billed.
   If you attach a billing account anyway, set a budget alert and a quota cap at 0.
2. Install the Google Cloud CLI, then in a terminal:
   `gcloud auth application-default login`
3. In `C:\dev\kpr-data\.env`: `GCP_PROJECT=<project id>`.
4. Tell the data session (or run `python -m kpr.collect.gdelt --dry-run 2019-06` in kpr-data):
   it records the estimate and the decision in decisions/data/.
Meanwhile: FNSPID (the fallback) is already downloaded (5.7 GB) and subset, so D8 can go
either way quickly once the estimate is known. Limit to keep in mind: FNSPID covers only US
tickers until 2023.
Urgency: high (gdelt_daily blocks the real panel D11, hence Q4/Q5 on real data)
