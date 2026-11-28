From: data
To: user (team, can be split between members)
Need: complete the European earnings dates in config/earnings_manual.csv (task D9, dossier §5.6)
Why: Europe has no free central source. yfinance pre-filled 735 dates (marked
`source: yfinance`, to confirm), but some companies are empty or partial. Earnings days are
excluded from the shock sample (dossier §7.4), so a missing date can turn a results day into
a fake "person" shock.
How:
- Open config/earnings_manual.csv. Rows with an empty `date` mark a company-year with no date:
  add one row per release of that year (date of the results publication, local date), with
  `source: manual` and your name in `checked_by`, then delete the placeholder row.
- For pre-filled yfinance rows, put your name in `checked_by` once confirmed (or fix the date).
- Then run `python -m kpr.collect.earnings` (in kpr-data): it rewrites the earnings table and
  lists the company-years that still look incomplete.
Priority (fully missing): LVMH (MC), Kering (KER), Stellantis (STLAM, FCA before 2021).
Partial: Danone (BN, 2015-2025), Nestlé (NESN), L'Oréal (OR), Renault (RNO 2015-2025),
Schneider (SU), Ubisoft (UBI), Unilever (ULVR), a few years of BioNTech, Spotify, Shopify, UBS.
Sources: investor-relations "financial calendar" pages and results press releases.
Urgency: medium (needed before the real-data panel, D11, and the events, Q4)
