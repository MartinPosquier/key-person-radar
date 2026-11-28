# Status: quant
Updated: 2026-11-28
Current: natural_experiments.csv aligned on the universe company_id values (branch quant/align-company-ids, stacked on Q13), PR open. Q13 numbers script PR open; docs/final/02 itself waits for week 4 (rule 11)
Done: Q1 (PR #3), Q2 (PR #6), Q3 (PR #8), Q4 (PR #13), Q5 (PR #14), Q6 (PR #16), Q7 (PR #17), Q8 (PR #18), Q9 (PR #19), Q10 (PR #20), Q11 (PR #21), Q12 (PR #22), all open
Blocked: PR #1 merged; merge order #3, #6, #8, #13, #14, #16, #17, #18, #19, #20, #21, #22, #28, then the id alignment. Real-data runs wait for D11 (panel) and M10 (scores)
Needs from others: data: MCD/CMG/UNH prices and pageviews, Mark Schneider in persons.yaml, unrelated-celebrity panel, 2026 case data (inbox/data); user: ff_factors contract (low urgency)
Next: real-data runs once D11 and M10 are merged (events table, event study, placebos, robustness, Q10-Q12), then in week 4 run scripts/quant_final_numbers.py --include-test after M12 and fill docs/final/02; answer inbox requests
