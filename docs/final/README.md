# Final documentation: how we sorted, split and interpreted the data

**Fill this folder in week 4 only**, once the pipeline has run end to end on real data
(after tasks D11, Q5, M10 and Q10 are merged). Before that, leave it alone: it is not a
deliverable of weeks 1–3.

This folder answers the questions a jury, a recruiter or an app visitor will ask at the end:

1. **How did we sort the data?** Which companies, people and days we kept, which we threw
   away, and why. Every filter comes with the number of rows it removed.
2. **How did we split the data?** Which periods train, validate and test the models, which
   events are held back for validation, and how we made sure the models never saw the future.
3. **How should each variable be read?** What a CEO's age, tenure, founder status or media
   attention tells us about the stock's reaction, and what it does **not** tell us.

## Files and owners

| File | Owner | Content |
| --- | --- | --- |
| `README.md` | user | This page |
| `01_data_collection_and_sorting.md` | data | Sources, universe, choice of people, cleaning, panel |
| `02_events_and_splits.md` | quant | From attention to shocks, filtering funnel, abnormal returns, splits |
| `03_feature_interpretation.md` | ml | Meaning of every variable (age, tenure, founder, attention, tone…), what the models found, how to read the scores |
| `numbers/data.json` | data | Every number quoted in file 01, written by `scripts/data_final_numbers.py` |
| `numbers/quant.json` | quant | Every number quoted in file 02, written by `scripts/quant_final_numbers.py` |
| `numbers/ml.json` | ml | Every number quoted in file 03, written by `scripts/ml_final_numbers.py` |

## Rules

- **No number typed by hand.** Each figure in the markdown files comes from that session's
  `numbers/<area>.json`, produced by a script that reads the real tables. A rerun of the
  pipeline then updates every figure at once, and nobody can quote a stale count.
- **Placeholders** look like `{{n_events_usable}}`. Each one is defined in the "Numbers to
  produce" table at the end of its file: name, meaning, and the exact table and filter it
  comes from.
- **Plain English, short sentences.** These pages are rendered in the app's Methodology
  page and reused in the final report. Write for a reader who knows finance but has not
  read the code.
- **Say what we cannot conclude.** Every interpretation states its limit (association, not
  causation; small samples; survivorship).

## How it is used

- **Web app:** the Methodology page (task M14) renders the three files in order and fills
  the placeholders from `numbers/*.json`.
- **Final report:** the "Data", "Method" and "Interpretation" chapters are adapted from
  these files.
- **Oral defence:** the filtering funnel (file 02) and the feature reading guide (file 03)
  are the two slides that answer most jury questions.

## End-of-project checklist (user)

- [ ] The three scripts ran on the final data and the JSON files are committed
- [ ] No `{{placeholder}}` remains unfilled in files 01–03
- [ ] Each filter in file 02 has its count and its reason
- [ ] Every variable used by the final model has a line in file 03
- [ ] The Methodology page of the app shows the three files correctly
