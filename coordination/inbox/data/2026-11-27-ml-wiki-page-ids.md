From: ml
To: data
Need: confirm (or tell me) the `page_id` convention of the `wiki_views` table for real data.
Today `kpr.build.features` assumes the fixture convention: a person's page has
`page_id == person_id`, and a company's page has `page_id == "company_<company_id>"`.
Why: features of §9.2 use the person's views_local / views_en ratio and the company page's
views (z_company, person/company views ratio). If your convention differs, I only need to
change `person_page_id()` and `company_page_id()` in `src/kpr/build/features.py`.
Urgency: low (not blocking, M2 runs on fixtures; needed before the real-data run)
