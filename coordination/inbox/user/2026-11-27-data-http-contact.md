From: data
To: user
Need: a generic team contact (e-mail or project URL) in `HTTP_CONTACT` of every worktree's
`.env`, and the same value in `.env.example` if you want it shared
Why: Wikimedia (pageviews, D4) and SEC EDGAR (D6) require a User-Agent with a contact
(dossier §5.2, §5.6). Without it the Wikipedia API already rate-limits us (HTTP 429 during
D2), and EDGAR may refuse requests. The collectors read `HTTP_CONTACT` automatically
(`kpr.collect.common.user_agent`).
Urgency: high for D4 and D6 (they run with a contact-less User-Agent meanwhile, slowly)
