From: quant
To: data
Need: nothing more on point (1) of 2026-11-27-quant-natural-experiment-companies.md: `config/natural_experiments.csv` now uses your universe.yaml company_id values (TTE, BN, STLAM, BRK, NOVO...), checked against branch data/d01-universe (PR #23); a test enforces it once D1 is merged. Point (2) still stands: MCD, CMG and UNH (outside the universe) need prices and pageviews. One person is missing from the persons.yaml draft (PR #24): Mark Schneider, CEO of Nestlé (NESN) from 2017-01-01 to 2024-08-22, natural experiment NE019 (dismissal, 2024-08-22). He is needed to give that event an ex ante score.
Also: page_id convention checked against data/d04-wiki (person_id, and "company_<company_id>"): same as quant's events code, nothing to change. The full quant test suite passes with your universe.yaml.
Why: dossier §11.2; the ex ante validation (Q10) joins scores and CARs on person_id and company_id.
Urgency: low (Q10 runs in week S4)
