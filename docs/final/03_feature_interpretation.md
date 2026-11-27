# 3. What each variable tells us, and how to read the scores

> Owner: ml session. Fill in week 4 (task M14). Numbers come from `numbers/ml.json`,
> written by `scripts/ml_final_numbers.py`. Guidance in *italics* is to be replaced.

**In one sentence:** *the stock reacts most to bad news about people who {{headline_finding}};
the CEO's age and tenure {{age_tenure_finding}}.*

## 3.1 How to read this page

- **Association, not causation.** "Older CEOs come with smaller reactions" means that, in
  our data and other things equal, the two go together. It does not mean that age causes
  the smaller reaction.
- **Other things equal.** Each effect is read with the other variables held fixed: the age
  effect is the effect of age for the same tenure, role, attention level and market context.
- **Three ways to see an effect,** and an effect is only claimed when they agree:
  1. sign and size of the Lasso coefficient (in points of CAR per standard deviation);
  2. SHAP dependence plot of the best tree model;
  3. stability: share of the {{param_stability_bootstrap}} bootstrap Lasso fits that keep
     the variable (reliable above {{param_stability_threshold}}).

## 3.2 Variable dictionary

Every variable used by the final model has one line. "Known at J0" must be yes for all of
them: no information from the future.

| Variable | Family | Definition | Source | Expected effect (hypothesis) | What we found | Stability |
| --- | --- | --- | --- | --- | --- | --- |
| `z_wiki` | A. Shock intensity | Wikipedia views vs previous 60 days, in standard deviations | `panel` | Bigger spike, bigger reaction | {{coef_z_wiki}} | {{stab_z_wiki}} |
| `z_news` | A | Article count vs normal | `panel` | Same | {{coef_z_news}} | {{stab_z_news}} |
| `tone_neg` | B. Tone | Negative part of the relative tone | `events` | Larger negative CAR | {{coef_tone_neg}} | {{stab_tone_neg}} |
| `tone_pos` | B | Positive part of the relative tone | `events` | Smaller positive CAR than `tone_neg` (asymmetry, H2) | {{coef_tone_pos}} | {{stab_tone_pos}} |
| `finbert_net` | B | FinBERT positive minus negative probability on headlines | `headlines_tone` | Same direction as tone | {{coef_finbert_net}} | {{stab_finbert_net}} |
| `ceo_age` | C. Person | Age in years on J0 | `mandates.birth_date` | *See §3.3* | {{coef_ceo_age}} | {{stab_ceo_age}} |
| `tenure_years` | C | Years in the role on J0 | `mandates.start` | *See §3.4* | {{coef_tenure}} | {{stab_tenure}} |
| `founder` | C | Founder still in office | `mandates.founder` | Larger reactions (H3) | {{coef_founder}} | {{stab_founder}} |
| `role_*` | C | CEO, CFO, chair, other | `mandates.role` | CEO above other roles | {{coef_role}} | {{stab_role}} |
| `attention_share` | C | Google Trends ratio person / company, previous year | `trends_ratio` | Larger when the person dominates the brand | {{coef_attention_share}} | {{stab_attention_share}} |
| `views_ratio` | C | Person / company Wikipedia views, previous year | `wiki_views` | Same | {{coef_views_ratio}} | {{stab_views_ratio}} |
| `specificity` | D. Specificity | Person z-score minus company z-score | `panel` | Larger when the news is really about the person | {{coef_specificity}} | {{stab_specificity}} |
| `log_mcap` | E. Context | Log market cap | `prices` | Smaller reactions for giants | {{coef_log_mcap}} | {{stab_log_mcap}} |
| `vol_20d` | E | Stock volatility over the 20 sessions before | `prices` | Larger reactions when already volatile | {{coef_vol_20d}} | {{stab_vol_20d}} |
| `vix` | E | Market fear index on J0 | `vix` | Larger reactions in stressed markets | {{coef_vix}} | {{stab_vix}} |
| `car_m5_m1` | E | Abnormal return over the 5 sessions before | `events` | Checks for news leaking before the spike | {{coef_car_pre}} | {{stab_car_pre}} |

*Add or remove lines to match the final feature list exactly.*

## 3.3 Focus: the CEO's age

**Why it could matter:**

- an older CEO may be seen as closer to retirement, so a scandal or health issue raises
  succession questions faster;
- a young CEO may carry more of the company's story, so the market reacts more to news
  about them.

The two stories predict opposite signs: the data decides.

**Pitfalls:**

- **Age and tenure move together.** Long-serving CEOs are older. Read age with tenure held
  fixed, and check their correlation ({{corr_age_tenure}}).
- **Founders are often older.** Hold `founder` fixed too.
- **Missing birth dates.** {{n_missing_birth_date}} people have no known birth date. Say
  how they were handled (excluded or imputed, and with what).

**Result:** *one paragraph with the Lasso coefficient, the SHAP dependence plot
(`reports/figures/shap_ceo_age.png`) and the stability share. Say plainly if the effect is
weak or unstable: "no reliable effect of age" is a valid finding.*

## 3.4 Focus: tenure in the company

**Why it could matter:**

- a long tenure can make the person and the company one in the market's eyes (strong
  key-person risk);
- or it can prove the company runs well around them (weaker risk).

**Pitfalls:**

- **Survivorship.** A long tenure means the CEO survived. Weak CEOs leave early, so
  long-tenure CEOs are a selected, successful group.
- **Succession resets.** At a change of CEO, tenure falls to zero. Check that the effect is
  not driven only by the first months of new CEOs.

**Result:** *as for age: coefficient, SHAP plot (`reports/figures/shap_tenure.png`),
stability, plain conclusion.*

## 3.5 Focus: founder, attention share and the bad-news asymmetry

- **Founder (H3):** *mean S− of founders vs non-founders ({{s_minus_founders}} vs
  {{s_minus_non_founders}}, Mann-Whitney p = {{p_founder}}).*
- **Attention share:** *does a person who takes a large share of the brand's attention move
  the stock more? Coefficient and SHAP interaction with tone.*
- **Asymmetry (H2):** *median S−/S+ ratio {{median_asymmetry}}; share of people with a ratio
  above 1: {{share_asymmetry_above_1}}.*

## 3.6 What feature selection kept

| Method (LAB 4) | Variables kept |
| --- | --- |
| Filter (mutual information + mRMR) | {{kept_filter}} |
| Wrapper (RFE) | {{kept_wrapper}} |
| Embedded (Lasso) | {{kept_lasso}} |
| Stability (bootstrap Lasso, ≥ {{param_stability_threshold}}) | {{kept_stability}} |

*Two sentences: which families survive everywhere, which never do.*

## 3.7 How to read the scores in the app

- **S− (0–10):** how much the stock usually falls, beyond the market, after a standard
  burst of bad news about this person. 10 is the top decile of 2015–2021.
- **S+ (0–10):** the same for good news.
- **S− / S+:** above 1, bad news weighs more than good news.
- **V (% of market value, with interval):** an estimate of the value lost if the person
  left suddenly, calibrated on real sudden departures. It is a model extrapolation, never
  a valuation.
- **Π (share of variance):** share of the stock's own yearly variance that happened on days
  of shocks about the person. It needs no model, so it is a sanity check for V.
- **"Low confidence" badge:** fewer than 8 usable shocks; the score leans on the average of
  similar people.

*Add two or three plain-English blurbs ready to paste into the app (one per score).*

## Numbers to produce (`scripts/ml_final_numbers.py` → `numbers/ml.json`)

| Placeholder | Meaning | Source |
| --- | --- | --- |
| `param_*` | Settings actually used | `config/params/ml.yaml` |
| `coef_*` | Lasso coefficient on the final training set, per standard deviation | Final Lasso model |
| `stab_*` | Share of bootstrap Lasso fits keeping the variable | Stability selection output |
| `corr_age_tenure`, `n_missing_birth_date` | Age diagnostics | `features`, `mandates` |
| `s_minus_founders`, `s_minus_non_founders`, `p_founder` | H3 test | `scores` joined with `mandates` |
| `median_asymmetry`, `share_asymmetry_above_1` | H2 | `scores.asymmetry` |
| `kept_*` | Selected variables per method | Feature selection outputs (task M9) |
| `headline_finding`, `age_tenure_finding` | One-line conclusions, written last | Written by hand from the numbers above |
