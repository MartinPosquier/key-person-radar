Date: 2026-11-27
Session: quant
Decision: tests of H2-H5 (Q11) on the scores table, one observation per person-company (mean over its scored years)
- H2: ratio of raw sensitivities s_minus_raw / s_plus_raw per person (both > 0); report share above 1, one-sided sign test (binomial) and one-sided Wilcoxon on log ratio. Raw rather than 0-10 scores because the published scale is shrunk and smoothed, which distorts ratios.
- H3: one-sided Mann-Whitney, founders (any founder mandate at that company in mandates) vs others, and media vs control (universe.yaml group), on s_minus, s_plus and v_pct. The "founder effect in the model" part of H3 is the ml session's coefficient (M8).
- H4: Spearman of the score in y vs y+1 for the same person and company; CEO successions from mandates (consecutive CEOs of a company); jump = |incoming score in the change year (or first scored year after) - outgoing score in the last scored year before|; one-sided Mann-Whitney jumps > ordinary yearly changes.
- H5: Jensen Huang vs Patrick Pouyanné in their latest common year, S-, S+ and V with the bootstrap intervals of the scores table; disjoint intervals -> "X higher", otherwise "no conclusion" (§11.5: "si les intervalles se chevauchent, on le dit"). Person ids follow the data session's convention; --h5 overrides them.
Why: §2 (table of hypotheses) and §11.5. One observation per person avoids counting long-serving leaders many times.
