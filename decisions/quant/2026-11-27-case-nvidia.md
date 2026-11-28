Date: 2026-11-27
Session: quant
Decision: the Nvidia 2026 case (Q12) is a reproducible command, not a hand calculation
- Date: the highest 2026 attention z-score of Jensen Huang's page (§11.6 step 1), overridable by a date sourced by hand (--date); the report states which.
- Reaction: market-model CAR[0, 2] and CAR[0, 5] (estimation window J-250..J-30 as in §7.5) next to the raw and index returns, for Nvidia and its semiconductor peers of the universe (AMD, INTC, ASML).
- Competing news: earnings within 5 days and the person's headlines of the day are listed automatically; export restrictions, customer news and sector moves are flagged for a manual check.
- Model: end-of-2025 scores; expected |CAR| for a negative shock = s_minus_raw (%, §10.6); verdict "consistent" if |actual| / expected is within [0.5, 2], otherwise under- or overestimated. The band is a named constant; the ml session may replace the expectation with method A's predicted CAR for this event's features once M10 exists.
- Data isolation: 2026 prices and pageviews are read from separate files (--views, --prices, --market), never from the shared study tables, so no 2026 row can reach events or models (period_split also refuses dates after 2025).
Why: §11.6 steps 1-5 and the risk table of §15 ("le présenter comme illustration, avec le CAR net du marché et la liste des nouvelles concurrentes").
