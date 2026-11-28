Date: 2026-11-27
Session: quant
Decision: shock detection and filters as implemented in kpr.build.events (task Q4)
Why (dossier §7.1-7.4), with the points the dossier leaves open:
- Clustering (§7.2, condition 4): a peak day opens a new event only if the person had no peak day in the 10 previous calendar days; chains of peak days less than 10 days apart stay one event, dated on the first day. Comparing with the previous peak day (not the previous event) is what "les jours de pic successifs sont regroupés" asks for, and it keeps reaction windows from overlapping.
- z-scores: z_wiki and z_news are read from the panel (data session). `attention_z` (log(1 + views), reference t-65..t-6, at least 20 reference days, as in the fixtures) is used for the company page and for robustness runs.
- Sign (§7.3): strict thresholds (tone_rel < -1 negative, > +1 positive); a missing tone counts as neutral (rare: at least 3 articles are required).
- Extreme market (§7.4): the index moves more than 4 % (absolute) on any session of the [0, 2] CAR window, not only on J0, because the market model is unreliable on every day it sums.
- Earnings (§7.4): each release date is aligned like J0 (first session on or after it), then the shock is excluded if J0 is within 1 session of it (parameter, 0/1/2 in robustness).
- Company dominance (§7.4): company page z-score on the shock day minus the person's z_wiki > 2. The company page is `company_<company_id>` in wiki_views (fixture convention; confirmation asked to data).
- Exclusions are flagged, never dropped, and every applicable reason is kept, joined by ";", so each filter can be counted (§7.4). Extra reason `missing_returns` when the target CAR cannot be computed for lack of price or index data.
- Markets: exchange calendar and index from universe.yaml; companies missing from it (fixtures) fall back to their price dates and the best-correlated index, with a warning.
Alternatives considered: clustering on events instead of peak days (rejected: a long affair would produce a new event every 11 days); dropping excluded shocks (rejected by §7.4).
