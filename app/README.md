# Web app: deployment notes

The app is a showcase: it reads precomputed files from `app/data/` and computes nothing
heavy (dossier §12.3). No API key, no secret.

## Refresh the data

```powershell
python -m kpr.scores.smoothing     # scores table (after sensitivity and value)
python -m kpr.scores.export_app    # writes app/data/*.parquet, meta.json, case_study.json
streamlit run app/Home.py          # check locally
```

`export_app` records the sanity checks of dossier §10.7 in `app/data/meta.json`
(control-group persons in the top 10, mean V by group, low-confidence count, high V with
Pi near zero) and logs a warning for each one that fails. Read them before publishing.
Until real results exist, `python scripts/ml_make_app_demo.py` fills `app/data/` with
synthetic data, and every page shows a "demo mode" banner.

## Deploy on Streamlit Community Cloud (free)

1. The GitHub repository must be public, with the exported files of `app/data/` committed
   (a few MB at most; check the sizes printed by `export_app`).
2. On share.streamlit.io: **New app**, repository `key-person-radar`, branch `main`,
   main file path `app/Home.py`, Python 3.12.
3. Dependencies come from `app/requirements.txt` (next to the entrypoint), not from the
   root `requirements.txt`, so torch and the ML stack are not installed.
4. After each data refresh: commit `app/data/`, push; the app redeploys by itself.
5. Put the public URL in the README (task M13) and in `docs/final`.

## Honest-display rules (dossier §12.4), enforced in `app/common.py`

- every V is shown with its 5th–95th percentile interval and the word "estimate";
- scores based on fewer than 8 shocks carry a "low confidence" badge;
- "Research project. Not investment advice." on every page (checked by the tests);
- capped scores (10) are flagged "(capped)";
- no causal wording: texts describe how the stock moved around the news.
