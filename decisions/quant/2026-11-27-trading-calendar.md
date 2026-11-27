Date: 2026-11-27
Session: quant
Decision: trading sessions from the `exchange_calendars` package; J0 = first session on or after the UTC shock day
Why: dossier §6.5. The package covers every exchange of the universe (XNYS, XNAS, XPAR, XETR, XAMS, XLON, XMIL, XSWX, XCSE, XDUB) with holidays, half days and daylight saving time, so session closes in UTC are exact. J0 follows rule 2 at day level (J0 = d if d is a session, else the next one); after-close news is covered by the J0..J0+2 window (rule 3). `reaction_session` handles precise GDELT timestamps for the before/after-close check of rule 5.
Alternatives considered: pandas_market_calendars (heavier, same data source); deriving sessions from the price table (breaks on missing quotes and cannot tell a holiday from a data gap); a hand-written holiday list (error-prone, no half days).
