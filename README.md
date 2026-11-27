# Key-Person Risk Radar

**How much of Nvidia is Jensen Huang?**

We measured how much 60 of the world's largest companies depend on a handful of people.
For each tracked CEO, founder or CFO and each year from 2015 to 2025, we estimate how strongly
the company's stock reacts to news about that person, without using any accounting data:

| Score | Meaning |
| --- | --- |
| **S⁻** (0–10) | Sensitivity of the stock to bad news about the person |
| **S⁺** (0–10) | Sensitivity of the stock to good news about the person |
| **V** (% of market value) | Share of market value that an abrupt departure would put at risk, an **estimate** shown with its 5th–95th percentile interval |

**Live app:** _coming soon_ (Streamlit Community Cloud)

> Research project (ESILV machine learning course). Not investment advice. The scores
> describe how stocks moved around news about a person; they do not prove the person caused
> those moves.

## The method in five lines

1. We detect **attention shocks**: days when a person's Wikipedia views and press coverage jump far above their usual level.
2. We measure the **tone** of that day's coverage (GDELT, then FinBERT on headlines) and set aside earnings days.
3. We compute the stock's **abnormal return** over three trading sessions, net of the market (event study).
4. **Machine learning models** learn how that reaction depends on the shock, its tone, the person and the market context, and must beat simple baselines out of sample.
5. From the models we derive S⁻, S⁺ and V for a standard shock, on a fixed 0–10 scale, smoothed over time and reset only at a succession.

## Install and run

Requires Python 3.12 and Git.

```powershell
git clone https://github.com/MartinPosquier/key-person-radar.git
cd key-person-radar
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt       # torch / transformers only needed for FinBERT
pip install -e . --no-deps
Copy-Item .env.example .env           # then set KPR_DATA_DIR to a folder outside the repo
```

Run the web app (it reads precomputed results from `app/data/`, no API key needed):

```powershell
streamlit run app/Home.py
```

Run the pipeline step by step (every command is listed by `python -m kpr.cli list`):

```powershell
python -m kpr.cli collect prices      # data collection
python -m kpr.cli build events        # attention shocks and abnormal returns
python -m kpr.cli build features      # feature matrix
python -m kpr.cli models baselines    # baselines and time-series cross-validation
```

Set `KPR_USE_FIXTURES=1` to run any step on the small synthetic dataset in `tests/fixtures/`.

## Repository layout

| Folder | Content |
| --- | --- |
| `src/kpr/collect/` | Data collectors: prices, Wikipedia, GDELT, EDGAR, Google Trends |
| `src/kpr/build/` | Cleaning, daily panel, abnormal returns, events, tone, features |
| `src/kpr/models/` | Baselines, models, cross-validation, metrics, explanations |
| `src/kpr/scores/` | From models to S⁻, S⁺ and V |
| `src/kpr/validation/` | Natural experiments, placebos, robustness |
| `app/` | Streamlit web app |
| `config/` | Universe, persons, parameters |
| `decisions/` | One file per method choice, with its reason |

## Data sources

Wikipedia pageviews (Wikimedia), GDELT, Yahoo Finance prices and indices, VIX, SEC EDGAR
filings, Google Trends, Wikidata. All free and public; see the Methodology page of the app.

## Team

ESILV A4 machine learning project, 2026.
