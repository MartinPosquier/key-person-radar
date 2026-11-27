Date: 2026-11-27
Session: quant
Decision: the events table keeps every detected shock (excluded ones flagged); natural experiments reserve every event of the same company within 10 calendar days of the news; split by J0 period (train up to 2021-12-31, including the few pre-2015 events; validation 2022-2023; test 2024-2025)
Why: §7.4 (excluded shocks are counted, not deleted), §11.2 ("exclure ces événements et leurs 10 jours voisins de toutes les données d'entraînement et de calibrage"), §9.5 (periods). Reserving by company rather than by person catches a second person's shock caused by the same news (a CFO quote on the CEO's arrest). J0 rather than date_utc decides the period because J0 is when the return is earned.
Shared table policy: `python -m kpr.build.events` writes processed/events only from real data with the default parameters. Fixture runs and robustness variants (`--set`) must use `--out`, so the shared table never holds synthetic or variant data.
Gap: the 10-session gap between train and validation (§9.5) comes with Q6 (splits.py).
