# 1. Data collection and sorting

> Owner: data session. Fill in week 4 (task D13). Numbers come from `numbers/data.json`,
> written by `scripts/data_final_numbers.py`. Guidance in *italics* is to be replaced.

**In one sentence:** *we started from {{n_candidate_persons}} candidate people at
{{n_companies}} companies and kept {{n_persons_final}} people whose public attention is
measurable every day from 2015 to 2025.*

## 1.1 Sources

*One row per source actually used. Say when it was downloaded and what it could not give us.*

| Source | What we took | Period | Rows | Downloaded on | Limits met |
| --- | --- | --- | --- | --- | --- |
| Wikipedia Pageviews | Daily views of people and companies | | {{rows_wiki_views}} | | |
| GDELT (BigQuery) or FNSPID | Daily article count and tone per person | | {{rows_gdelt_daily}} | | |
| yfinance | Adjusted prices, indices, VIX | | {{rows_prices}} | | |
| Wikidata | Mandates, founders, birth dates | | {{rows_mandates}} | | |
| SEC EDGAR + manual EU file | Earnings dates | | {{rows_earnings}} | | |
| Google Trends | Person / company attention ratio | | {{rows_trends_ratio}} | | |

*Write here which source was dropped or replaced during the project and why (for example
GDELT replaced by FNSPID because of the BigQuery quota).*

## 1.2 Universe of companies

*How the companies were chosen (dossier §4.1) and how many of each kind we ended with.*

| Group | Companies | Why this group exists |
| --- | --- | --- |
| Media-exposed leaders | {{n_companies_media}} | Where key-person risk is expected |
| Low-profile leaders (control) | {{n_companies_control}} | Checks that the method finds little where little is expected |
| US / Europe | {{n_companies_us}} / {{n_companies_eu}} | |
| Listed after 2015 (shorter history) | {{n_companies_late_listing}} | |

**Survivorship:** *only companies still listed today are included. Name the famous cases
left out (Wirecard, Credit Suisse, Twitter) and say how this may bias the scores (probably
downwards: the worst key-person failures disappeared from the market).*

## 1.3 How we chose the people we follow

This is the first sorting step and the one the jury will ask about. Show it as a funnel.

| Step | Rule (dossier §4.4) | People left | Removed |
| --- | --- | --- | --- |
| 0. Candidates | CEOs, founders, chairs from Wikidata + manual additions | {{n_candidate_persons}} | |
| 1. Relative attention | Google Trends ratio person / company ≥ {{param_min_trends_ratio}} | {{n_after_trends}} | {{n_removed_trends}} |
| 2. Wikipedia volume | Median daily views ≥ {{param_min_median_views}} | {{n_after_views}} | {{n_removed_views}} |
| 3. Enough shocks | At least {{param_min_shocks}} usable shocks over the period | {{n_after_shocks}} | {{n_removed_shocks}} |
| 4. Manual check | Homonyms, poor pages, wrong mandate dates | {{n_persons_final}} | {{n_removed_manual}} |

*Then two or three sentences: which kinds of people were removed most often (CFOs with
little attention? recent CEOs?), and what that means for the results: the scores describe
people the public pays attention to, not every executive.*

**Roles kept:** CEO {{n_role_ceo}}, CFO {{n_role_cfo}}, chair {{n_role_chair}}, other
{{n_role_other}}. **Founders still in office:** {{n_founders}}.
**Birth date known:** {{n_birth_date_known}} of {{n_persons_final}} (needed for the age variable).

## 1.4 Cleaning

| Rule (dossier §6.4) | Rows affected | What we did |
| --- | --- | --- |
| Missing pageview days | {{n_missing_views}} | Left missing, never set to zero |
| Pageview spikes with no matching press coverage (likely bots) | {{n_bot_spikes}} | Removed from shock detection |
| Duplicate articles (same title, same day) | {{n_duplicate_articles}} | Deduplicated |
| Days with fewer than {{param_min_articles_for_tone}} articles | {{n_low_article_days}} | Tone left missing |
| Extreme daily returns checked by hand | {{n_extreme_returns_checked}} | *Kept / corrected (say which)* |
| Missing earnings dates filled by hand | {{n_earnings_manual}} | Entered from investor-relations calendars |

## 1.5 Time alignment

*Two sentences: attention data is per UTC day, prices close in local time; a shock on UTC
day d is matched with the first trading session that starts on or after d, and the 3-day
window [J0, J+2] absorbs news released after the close (dossier §6.5).*

## 1.6 The daily panel

The panel has one row per (person, trading day): {{n_panel_rows}} rows, {{n_persons_final}}
people, {{n_trading_days}} trading days. Share of rows with complete attention, tone and
return data: {{share_panel_complete}}.

## Numbers to produce (`scripts/data_final_numbers.py` → `numbers/data.json`)

| Placeholder | Meaning | Source |
| --- | --- | --- |
| `n_companies`, `n_companies_media`, `n_companies_control`, `n_companies_us`, `n_companies_eu` | Universe counts | `config/universe.yaml` |
| `n_companies_late_listing` | First price after 2015-01-01 | `prices`, min date per company |
| `n_candidate_persons` | Candidates before any filter | Wikidata extract + manual additions |
| `n_after_trends`, `n_after_views`, `n_after_shocks`, `n_persons_final` and the matching `n_removed_*` | Selection funnel | Selection log written by the selection step |
| `param_*` | Thresholds actually used | `config/params/data.yaml` |
| `n_role_*`, `n_founders`, `n_birth_date_known` | Final people | `mandates` restricted to final people |
| `rows_*` | Rows per interim table | `interim/*.parquet` |
| `n_missing_views`, `n_bot_spikes`, `n_duplicate_articles`, `n_low_article_days`, `n_extreme_returns_checked`, `n_earnings_manual` | Cleaning counts | Cleaning log of `build/clean.py` |
| `n_panel_rows`, `n_trading_days`, `share_panel_complete` | Panel size | `processed/panel/panel.parquet` |
