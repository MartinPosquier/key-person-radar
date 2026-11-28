Date: 2026-11-27
Session: data
Decision: broad list of persons (dossier §4.4 step 1) = every CEO in office at some point
in 2014-2025 for the 60 companies, plus the other candidates named in dossier §4.2
(founders still active such as Page, Brin, Ellison, Thiel; executive chairs such as Bezos and
Hastings; designated successors such as Greg Abel). Non-executive chairs that Wikidata lists
are not added: they are reported in the check list instead. Result: 148 mandates, 134 persons.
- `end` is the last day in the role (inclusive); `null` = still in office on the last
  information date. Mandates that end after 2025 keep their real date (out of the period).
- Roles: CEO, chair, other (executive chair is `chair`, except Iger 2020-2021 = `other`
  because he was no longer CEO but kept an operational role), founder flag separate.
- Overlapping CEO mandates are allowed only with `co_ceo: true` (Netflix, Oracle, Salesforce,
  SAP, Deutsche Bank, Intel interim). A test enforces it (dossier §6.7).
- FCA CEOs (Marchionne, Manley) are attached to `STLAM` because the yfinance history before
  2021 is FCA's.
- Persons without an English Wikipedia page are kept (they will fail the 200 views/day
  filter of §4.4 step 3) so that the broad list stays complete.
- Wikidata is read through the `wbgetentities` API, not SPARQL: the query service timed out.
Why: dossier §4.4, §5.6, §6.7. Wikidata agrees (within 31 days) with only 56 of 148 draft
mandates; 39 differ (mostly year-only precision in Wikidata) and 53 are absent. Every
non-agreeing mandate is listed for a human check in
coordination/inbox/user/2026-11-27-data-persons-check.md.
Alternatives considered: taking Wikidata dates as they are (rejected: many are year-only or
wrong, e.g. Visa's Kelly as CEO from 2019, Schultz 1982-2017 in one block).
