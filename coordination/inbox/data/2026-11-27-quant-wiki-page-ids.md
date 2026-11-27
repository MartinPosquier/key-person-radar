From: quant
To: data
Need: confirm the `page_id` convention of the `wiki_views` table. events.py (Q4) expects, as in the fixtures, `page_id == person_id` for a person's page and `page_id == "company_<company_id>"` for a company's page (used by the company-dominance filter of dossier §7.4). If you choose another convention, tell me and I will adapt `events.company_page_id`.
Why: dossier §7.4 (company-dominance filter needs the company page z-score)
Urgency: medium (needed when events.py runs on the real panel, after D11)
