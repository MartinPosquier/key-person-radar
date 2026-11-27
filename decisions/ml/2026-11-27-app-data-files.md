Date: 2026-11-27
Session: ml
Decision: the Streamlit app reads four Parquet files and two JSON files from `app/data/`:
`scores` (the `scores` table), `events` (the `events` table plus `car_pred`), `daily`
(person_id, date, company_id, views, close_adj, ar), `entities` (display names, role,
region, sector, group), `meta.json` (data origin, `demo` flag, export date) and
`case_study.json` (Nvidia 2026 case). Until real scores exist, `scripts/ml_make_app_demo.py`
builds these files from the test fixtures, and every page shows a "demo mode" banner.
Why: dossier §12.3 asks for precomputed files only (scores, events, daily series). A fourth
file, `entities`, carries names and the filters of the Leaderboard (region, sector, role),
which are not in the frozen `scores` table; keeping it separate avoids a contract change.
The company score follows §3.7 (dependence index = max of V, upper bound = sum of V). The
honest-display rules of §12.4 live in `app/common.py`: V always shown with its 5th–95th
percentile interval and "estimate", "low confidence" under 8 shocks, disclaimer on every
page, capped scores flagged (§3.3).
Alternatives considered: joining names into the scores file (rejected: the app file would
no longer match the frozen schema, which makes the real export harder to check); reading
the shared tables directly from KPR_DATA_DIR (rejected: Streamlit Community Cloud has no
access to it, §12.3).
