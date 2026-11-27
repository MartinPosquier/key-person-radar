From: quant
To: data
Need: a small panel of 20-30 unrelated celebrities (athletes, actors, musicians with no corporate role), with the same columns as the `panel` table (person_id, date, company_id may be empty or "NONE", views, z_wiki, n_articles, z_news, tone_rel, ret NaN, earnings_window False, vix), 2015-2025, saved as a parquet file anywhere under interim/ (it is not a frozen table; just tell me the path).
Why: placebo "personnes sans lien" of dossier §11.3 (task Q8). kpr.validation.placebo already takes it through `--unrelated-panel`.
Urgency: low (week S3; the other three placebos run without it)
