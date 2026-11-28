# 2. From attention to events, and how the data was split

> Owner: quant session. Fill in week 4 (task Q13). Numbers come from `numbers/quant.json`,
> written by `scripts/quant_final_numbers.py`. Guidance in *italics* is to be replaced.

**In one sentence:** *out of {{n_shocks_raw}} attention spikes detected, {{n_events_usable}}
were clean enough to measure a stock reaction; the models learned on 2015–2021, were tuned
on 2022–2023 and were judged once on 2024–2025.*

## 2.1 What counts as a shock

A shock is a day when attention on a person jumps far above its recent level (dossier §7.2):

- Wikipedia views more than {{param_z_wiki_min}} standard deviations above the previous
  60 days (with a 5-day gap so a spike does not inflate its own baseline);
- confirmed by the press: article count more than {{param_z_news_min}} standard deviations
  above normal, and at least {{param_min_articles}} articles that day;
- consecutive spike days within {{param_cluster_days}} days count as one event.

**Sign of the shock:** tone of the day's articles relative to the person's usual coverage.
Below −{{param_tone_threshold}} standard deviation: negative. Above +{{param_tone_threshold}}:
positive. In between: neutral.

## 2.2 The filtering funnel

This is the core of how we sorted the data. Each step removes shocks where we could not
tell whether the person, rather than something else, moved the stock.

| Step | Rule (dossier §7.4) | Shocks left | Removed | Why it matters |
| --- | --- | --- | --- | --- |
| 0. Detected | Rule of §2.1 | {{n_shocks_raw}} | | |
| 1. Earnings | Not within ±{{param_earnings_window}} session of a quarterly release | {{n_after_earnings}} | {{n_removed_earnings}} | Otherwise we would measure the results, not the person |
| 2. Company-driven news | Company page spike not much larger than the person's | {{n_after_company}} | {{n_removed_company}} | Product recall or accident where the CEO is only quoted |
| 3. Extreme market days | Market move under {{param_extreme_market}} in absolute value | {{n_after_market}} | {{n_removed_market}} | The market model is unreliable on crash days |
| 4. Enough history | At least {{param_min_estimation_obs}} sessions to estimate beta | {{n_events_usable}} | {{n_removed_history}} | Right after an IPO, beta is unknown |

*Then say which filter removed the most, and whether filtering changed the mix of shocks
(for example: earnings removed mostly positive shocks). Show the split below.*

| | Negative | Neutral | Positive | Total |
| --- | --- | --- | --- | --- |
| Media-exposed group | {{n_media_neg}} | {{n_media_neu}} | {{n_media_pos}} | {{n_media}} |
| Control group | {{n_control_neg}} | {{n_control_neu}} | {{n_control_pos}} | {{n_control}} |

## 2.3 Measuring the reaction

*Two sentences: abnormal return = stock return minus what the market explains, with a beta
estimated from 250 to 30 sessions before the shock; the target is the sum over J0 to J+2
(CAR). Robustness used the market-adjusted model, Fama-French 3 factors and windows [0,1],
[0,5], [−1,1] (dossier §7.5).*

**First result, before any machine learning** (dossier §7.6):

| Group | Mean CAR[0,2] | t-stat | Events |
| --- | --- | --- | --- |
| Negative shocks | {{car_mean_neg}} | {{car_t_neg}} | {{n_neg}} |
| Neutral shocks | {{car_mean_neu}} | {{car_t_neu}} | {{n_neu}} |
| Positive shocks | {{car_mean_pos}} | {{car_t_pos}} | {{n_pos}} |

*Add the CAR curve from J−10 to J+10 (`reports/figures/car_curve.png`) and one sentence on
what it shows: reaction on J0? drift before? reversal after?*

## 2.4 How the data was split

**Why not a random split:** a random split would let the model learn from 2024 events to
predict 2019 events. In finance that is looking into the future and gives scores that
collapse in real use. All splits follow time.

| Set | Period | Events | Used for |
| --- | --- | --- | --- |
| Train | 2015–2021 | {{n_train}} | Fitting models and tuning hyperparameters by time-series cross-validation |
| Validation | 2022–2023 | {{n_validation}} | Choosing the final model, calibrating λ for the value share V |
| Test | 2024–2025 | {{n_test}} | Final evaluation, run **once** |
| Reserved | Natural experiments, any year | {{n_reserved}} | Validating the scores only, never seen in training |

**Safeguards against leakage:**

- a gap of {{param_gap_sessions}} sessions between train and validation folds, so an event
  window never straddles the boundary;
- yearly attention ratios use the **previous** year only;
- scaling and feature selection are fitted inside each training fold (LAB 4, Q5);
- natural experiments and their 10 neighbouring days are removed from all training data;
- the test set was opened on {{test_opened_on}}, after the model was frozen.

**Cross-validation folds inside 2015–2021** (expanding window):

| Fold | Train years | Validation year | Train events | Validation events |
| --- | --- | --- | --- | --- |
| 1 | 2015–2017 | 2018 | | |
| 2 | 2015–2018 | 2019 | | |
| 3 | 2015–2019 | 2020 | | |
| 4 | 2015–2020 | 2021 | | |

**Are the sets comparable?** *Compare the three sets on share of negative shocks, sectors,
regions and mean VIX. If 2020 (Covid) or 2024–2025 look very different, say it: it explains
part of any drop in test performance.*

| | Train | Validation | Test |
| --- | --- | --- | --- |
| Share negative shocks | {{share_neg_train}} | {{share_neg_val}} | {{share_neg_test}} |
| Share US companies | {{share_us_train}} | {{share_us_val}} | {{share_us_test}} |
| Mean VIX on event days | {{vix_train}} | {{vix_val}} | {{vix_test}} |

## 2.5 Natural experiments kept aside

*List every reserved event (person, company, date, kind, wanted by the company or not) and
its real CAR. Say how many came from outside the 60-company universe.*

## Numbers to produce (`scripts/quant_final_numbers.py` → `numbers/quant.json`)

| Placeholder | Meaning | Source |
| --- | --- | --- |
| `param_*` | Thresholds actually used | `config/params/quant.yaml` |
| `n_shocks_raw`, `n_after_*`, `n_removed_*`, `n_events_usable` | Filtering funnel | `events`, columns `excluded` and `exclusion_reason` |
| `n_media_*`, `n_control_*` | Usable events by group and sign | `events` joined with `config/universe.yaml` |
| `car_mean_*`, `car_t_*`, `n_neg`, `n_neu`, `n_pos` | Event study | `events` where `excluded` is false and `split` is not reserved |
| `n_train`, `n_validation`, `n_test`, `n_reserved` | Split sizes | `events.split` |
| `share_*`, `vix_*` | Comparability of the sets | `events` joined with `universe.yaml` and `vix` |
| `test_opened_on` | Date of the single test run | `decisions/ml/` entry for task M12 |
