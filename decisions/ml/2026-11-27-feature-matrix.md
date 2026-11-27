Date: 2026-11-27
Session: ml
Decision: the `features` table holds only numeric columns (float), one row per event of
`events` (excluded and reserved events included; models filter them). Categorical
variables (role, region, sector) are one-hot encoded on fixed category lists taken from
config (`ml.yaml` roles and regions, sectors of `universe.yaml`), with an `_other` column.
No scaling in the table: standardisation happens inside each scikit-learn pipeline, fitted
on the training fold only.
Timing rules (no look-ahead, dossier §9.2): attention and tone from the shock day
(`date_utc`); volatility over the 20 sessions strictly before J0; market cap at the last
session before J0; VIX and market return of J0 (explicitly allowed by §9.2); Trends ratio of
the previous calendar year; person/company views ratio over the 365 days before the shock;
peak duration = consecutive days up to and including the shock day with z_wiki > 2 (days
after the shock would leak the reaction window). A test scrambles every input after J0 and
checks that the features of the event do not move.
Specificity (family D): z_company is recomputed from the company page's views with the
§7.1 window (60 days, 5-day gap, read from quant.yaml), and z_gap = z_wiki - z_company.
Why: dossier §9.2; one-hot on fixed lists uses no data, so it is leak-free and lets every
model read the table directly.
Alternatives considered: the "share of the day's company articles that cite the person"
(§9.2, family D) is not built: no table of the frozen contracts counts company articles.
It can be added if the data session ever produces company-level GDELT counts.
Peak duration over the whole cluster (rejected: it uses days after J0).
