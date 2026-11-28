Date: 2026-11-27
Session: data
Decision: Wikipedia pageviews (task D4)
- Agent `user`, access `all-access`, daily, 2015-07-01 (start of the API) to 2025-12-31.
- page_id = person_id for persons, `company_<company_id>` for companies (as in the fixtures).
- English page plus the local-language page when listed (fr, de, it, nl, da);
  views_total = views_en + views_local. views_local = 0 when there is no local page (as in
  the fixtures); a day without any data stays missing (dossier §6.4: never zero).
- Renamed pages: views made before a move stay on the old title. We add the views of the
  former titles, i.e. the English redirects in "Category:Redirects from moves" (77 of 2,459
  redirects, e.g. "Facebook, Inc." for Meta, "Novo Nordisk" for Novo, "Jen-Hsun Huang").
  Other redirects (typos, variants) carry little traffic; fetching all of them multiplied
  the requests by 15 and triggered Wikimedia's rate limit (HTTP 429, Retry-After ~18 s).
  Local-language wikis have no such category: canonical titles only there.
- The raw cache honors Retry-After. Collection is slow without a contact in the User-Agent
  (HTTP_CONTACT request in coordination/inbox/user/).
Why: dossier §5.2, §4.5, §6.4.
Alternatives considered: summing all redirects (rejected: rate limits, marginal gain);
canonical titles only (rejected: breaks the series of renamed pages, e.g. Meta before 2021).
