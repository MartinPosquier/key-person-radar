# Parallel sessions playbook

Three Claude Code sessions (`data`, `quant`, `ml`) work at the same time, each in its own git
worktree and on its own branches. They never conflict because:

1. every file and data folder has **exactly one owner** (§3, §4);
2. the files everyone depends on are written once, in Phase 0, and then **frozen** (§2, §6);
3. downstream sessions build on **synthetic fixtures** until upstream work is merged (§7);
4. sessions talk only through **new files** in `coordination/` (§9);
5. only the **user merges**, in a fixed order (§8, §10).

```
Phase 0 (main checkout, alone)  ──►  merged into main
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
  kpr-data  (data/*)            kpr-quant (quant/*)             kpr-ml (ml/*)
  collect, clean, panel         calendar, returns, events,      features, models, scores,
                                validation                      tone, app
        │  PRs                         │  PRs                         │  PRs
        └──────────────► user merges into main, every day ◄───────────┘
                         weekly integration run + tag (§10)
```

---

## 1. Setup, done once by the user

Prerequisites: Git, GitHub CLI (`gh auth login` done), Python 3.12 (`py -3.12`). Python 3.14
is too recent for some libraries (torch).

**Keep the repo out of OneDrive.** OneDrive syncing `.git` can corrupt it. Use `C:\dev\`.

```powershell
mkdir C:\dev
cd C:\dev
gh repo create key-person-radar --public --clone
mkdir C:\dev\kpr-data-store        # shared KPR_DATA_DIR, never inside a worktree
```

Copy `CLAUDE.md` and `PARALLEL_SESSIONS.md` from the planning folder into
`C:\dev\key-person-radar`, then run Phase 0 (§2) in that folder.

On GitHub: **Settings → Collaborators** (add the team), then **Settings → Branches →
Add rule** for `main`, with "Require a pull request before merging" and "Require status
checks to pass". This is what makes "never push to main" impossible to break by accident.

---

## 2. Phase 0: bootstrap (one session, alone, in `key-person-radar`)

No parallel work starts before this PR is merged. Branch: `chore/bootstrap`.

- [ ] Folder tree of dossier §13.1, with empty `__init__.py` files
- [ ] `pyproject.toml`: src layout, package `kpr`, **no dependencies listed** (they live in `requirements/`)
- [ ] `requirements.txt` containing only `-r` lines for `requirements/base.txt`, `data.txt`,
      `quant.txt`, `ml.txt`, `nlp.txt`, `dev.txt`; and those six files with bounded version ranges
      (`nlp.txt` holds torch and transformers, kept apart so CI does not install them)
- [ ] `.gitignore` (`.venv/`, `.env`, `data/`, `__pycache__/`, `.ipynb_checkpoints/`, `*.parquet` outside `tests/fixtures` and `app/data`)
- [ ] `.env.example` with `KPR_DATA_DIR=C:\dev\kpr-data-store`, `GCP_PROJECT=`, `HTTP_CONTACT=`
- [ ] `src/kpr/schemas.py`: every table of §6 below, with column names, dtypes, keys, plus
      a `validate(df, table_name)` function that raises on mismatch
- [ ] `src/kpr/utils/io.py`: `data_path(area, name)`, `read_table(name)`, `write_table(df, name)`.
      `write_table` validates against the schema, writes to a temp file, then renames it (atomic, so
      a reader in another worktree never sees a half-written file), and refuses to write outside
      the caller's area
- [ ] `src/kpr/cli.py`: a frozen table mapping every planned command to `"kpr.<pkg>.<module>:main"`
      (all modules of §13.1); each target module exists with a `main()` that raises `NotImplementedError`
- [ ] `config/params/data.yaml`, `quant.yaml`, `ml.yaml`, with the default values from the dossier
      (thresholds of §7.2, windows of §7.5, splits of §9.5, scale and smoothing of §3)
- [ ] `scripts/make_fixtures.py`: synthetic data for 5 fake persons and 2 years, one file per table
      of §6, valid against `schemas.py`, with a few planted shocks of known size; output in `tests/fixtures/`
- [ ] `tests/test_schemas.py`: every fixture passes `validate`
- [ ] `.github/workflows/ci.yml`: install, `ruff check .`, `pytest` on every PR
- [ ] `.github/pull_request_template.md` (what, why, how tested, dossier §)
- [ ] Empty folders with `.gitkeep`: `coordination/inbox/{data,quant,ml,user}/`, `coordination/status/`,
      `decisions/{data,quant,ml}/`
- [ ] `coordination/status/{data,quant,ml}.md`, initialised from the template in §9
- [ ] `README.md` placeholder (title, one-line pitch, "work in progress")
- [ ] GitHub labels `data`, `quant`, `ml`, `contract` (`gh label create data`, etc.)

After merge, the user creates the three worktrees (§5).

---

## 3. File ownership

A session edits only what it owns. "Frozen" means: changed only through a contract PR (§6).

| Path | Owner |
| --- | --- |
| `src/kpr/collect/**` | data |
| `src/kpr/build/clean.py`, `src/kpr/build/panel.py` | data |
| `config/universe.yaml`, `config/persons.yaml`, `config/params/data.yaml` | data |
| `requirements/data.txt`, `tests/data/**`, `notebooks/01_*`, `scripts/data_*`, `decisions/data/**` | data |
| `src/kpr/utils/calendar.py` | quant |
| `src/kpr/build/returns.py`, `src/kpr/build/events.py`, `src/kpr/build/splits.py` | quant |
| `src/kpr/validation/**` | quant |
| `config/params/quant.yaml`, `config/natural_experiments.csv` | quant |
| `requirements/quant.txt`, `tests/quant/**`, `notebooks/02_*`, `scripts/quant_*`, `decisions/quant/**` | quant |
| `src/kpr/build/tone.py`, `src/kpr/build/features.py` | ml |
| `src/kpr/models/**`, `src/kpr/scores/**` | ml |
| `app/**`, `README.md` | ml |
| `config/params/ml.yaml`, `config/score_scale.yaml` | ml |
| `requirements/ml.txt`, `requirements/nlp.txt`, `tests/ml/**`, `notebooks/03_*`, `notebooks/04_*`, `scripts/ml_*`, `decisions/ml/**` | ml |
| `coordination/status/<s>.md`, `coordination/inbox/<s>/**` (others may only **add** files there) | session `<s>` |
| `src/kpr/schemas.py`, `src/kpr/utils/io.py`, `src/kpr/utils/logging.py`, `src/kpr/cli.py` | frozen |
| `pyproject.toml`, `requirements.txt`, `requirements/base.txt`, `requirements/dev.txt` | frozen |
| `.github/**`, `.gitignore`, `.env.example`, `scripts/make_fixtures.py`, `tests/fixtures/**`, `tests/test_schemas.py` | frozen |
| `CLAUDE.md`, `PARALLEL_SESSIONS.md` | user only |

**Running a module:** each module is runnable as `python -m kpr.<pkg>.<module>`, so a session
never needs to edit `cli.py` to try its code.

---

## 4. Shared data directory

All worktrees point `KPR_DATA_DIR` (in their own `.env`) to the **same** folder,
`C:\dev\kpr-data-store`. Each subfolder has a single writer. `io.write_table` enforces this.

| Subfolder | Writer | Readers |
| --- | --- | --- |
| `raw/<source>/` | data | data |
| `interim/` (prices, market, vix, wiki_views, gdelt_daily, trends_ratio, earnings, mandates, headlines) | data | all |
| `processed/panel/` | data | quant, ml |
| `processed/events/`, `processed/validation/` | quant | ml |
| `processed/tone/`, `processed/features/`, `processed/models/`, `processed/scores/` | ml | quant (validation), app export |

Never delete or overwrite another session's files. To recompute something you do not own,
write an inbox request (§9).

---

## 5. Creating the worktrees and launching the 3 sessions

After Phase 0 is merged, run once, from `C:\dev\key-person-radar`:

```powershell
git switch main
git pull
git worktree add ..\kpr-data  -b data/universe       origin/main
git worktree add ..\kpr-quant -b quant/calendar      origin/main
git worktree add ..\kpr-ml    -b ml/app-skeleton     origin/main
```

In **each** of the three folders, once:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt   # data and quant may skip nlp.txt (torch) if disk is short
pip install -e . --no-deps
Copy-Item .env.example .env       # then set KPR_DATA_DIR=C:\dev\kpr-data-store
```

Every worktree has its own `.venv`, because an editable install points to one folder's `src`.

Then open three terminal tabs, one per folder, run `claude` in each, and give the opening prompt:

- **kpr-data:** « Tu es la session data. Lis CLAUDE.md puis PARALLEL_SESSIONS.md, puis traite ta file §7.1 dans l'ordre, en commençant par la première tâche non faite d'après coordination/status/data.md. Une branche et une PR par tâche, tu ne fusionnes rien. »
- **kpr-quant:** same prompt with "quant" and §7.2.
- **kpr-ml:** same prompt with "ml" and §7.3.

**Working without interruptions.** The sessions are told to decide from the dossier, to
build on fixtures when blocked, and to queue questions as files in `coordination/inbox/user/`
instead of stopping. Which permission mode the sessions run in is the user's choice. The
GitHub branch protection on `main` stays the real safety net, whatever the mode.

When a worktree is no longer needed: `git worktree remove ..\kpr-data` (from the main checkout).

---

## 6. Data contracts (frozen in `src/kpr/schemas.py`)

Column lists follow dossier §6.3, §7.7 and §10.6. Keys in bold.

| Table | Producer | Columns |
| --- | --- | --- |
| prices | data | **company_id, date**, close_adj, volume, shares_out, ret |
| market | data | **index_id, date**, close, ret |
| vix | data | **date**, vix |
| wiki_views | data | **page_id, date**, views_en, views_local, views_total |
| gdelt_daily | data | **person_id, date**, n_articles, tone_mean, tone_neg_share, tone_std |
| trends_ratio | data | **person_id, year**, ratio_person_company |
| earnings | data | **company_id, date**, quarter, source |
| mandates | data | **person_id, company_id, role, start**, end, founder |
| headlines | data | **person_id, date, headline_id**, title, source |
| panel | data | **person_id, date**, company_id, views, z_wiki, n_articles, z_news, tone_rel, ret, earnings_window, vix |
| events | quant | **event_id**, person_id, company_id, date_utc, j0, z_wiki, z_news, n_articles, tone_rel, tone_neg_share, sign, car_0_2, car_m5_m1, car_0_1, car_0_5, excluded, exclusion_reason, is_natural_experiment, split |
| headlines_tone | ml | **person_id, date**, finbert_pos, finbert_neg, finbert_net, n_headlines |
| features | ml | **event_id**, one column per feature of §9.2 |
| scores | ml | **person_id, company_id, year**, s_minus_raw, s_plus_raw, s_minus, s_plus, s_minus_lo, s_minus_hi, s_plus_lo, s_plus_hi, asymmetry, v_pct, v_usd, v_lo, v_hi, pi_share, n_events, method_agreement, succession_flag |

**Changing a contract:**

1. The session that needs it writes `coordination/inbox/user/<date>-contract-<slug>.md`: the change, the reason, who is affected.
2. The user (or a session in the main checkout) opens a `contract/<slug>` PR that updates `schemas.py`, the fixtures and `test_schemas.py` together.
3. That PR is merged **before** any PR that relies on it. Every session then rebases on `origin/main`.

Adding a column at the end is cheap. Renaming or removing one is expensive: avoid it.

---

## 7. Task queues

Each task is one branch and one PR. "Real data" steps wait until the upstream PR named in
"Needs" is merged. Until then, build and test on the fixtures.

### 7.1 Session data

| # | Week | Task | Dossier | Needs |
| --- | --- | --- | --- | --- |
| D1 | S1 | `universe.yaml`: 60 companies, every ticker checked on yfinance | §4.2 | none |
| D2 | S1 | Wikidata query for mandates, then draft `persons.yaml` and a list of entries for the user to check by hand (in `coordination/inbox/user/`) | §4.4, §5.6 | D1 |
| D3 | S1 | Collector for prices, indices and VIX, 2014–2025, cached | §5.4 | D1 |
| D4 | S1 | Wikipedia pageviews collector, English + local language, user agent from `.env` | §5.2 | D2 |
| D5 | S1 | GDELT dry run on 1 month, extrapolated to 10 years. Rule: estimate under 50 % of the free quota → GDELT, otherwise FNSPID. Record the decision | §5.3 | D2 |
| D6 | S1 | EDGAR earnings dates (8-K item 2.02) for US companies | §5.6 | D1 |
| D7 | S1 | Google Trends ratios, best effort, cached, stop on repeated 429 errors | §5.5 | D2 |
| D8 | S2 | Full collection, and `gdelt_daily` (or FNSPID headlines) for all persons | §5.3 | D4, D5 |
| D9 | S2 | Template CSV for EU earnings dates, for the team to fill by hand; loader | §5.6 | D1 |
| D10 | S2 | `clean.py` and automatic quality checks | §6.4, §6.7 | D8 |
| D11 | S2 | `panel.py`, written to `processed/panel/` | §6.6 | D10, Q1 merged |
| D12 | S3 | `headlines` table for shock days and their neighbours | §8.3 | Q5 merged |
| D13 | S4 | Data documentation (sources, dates, limits) as a markdown file for the report | §5 | none |

### 7.2 Session quant

| # | Week | Task | Dossier | Needs |
| --- | --- | --- | --- | --- |
| Q1 | S1 | `calendar.py`: trading sessions per exchange, UTC to J0 alignment, unit tests (weekend, holiday, after-close news) | §6.5 | none |
| Q2 | S1 | `returns.py`: market model, market-adjusted, Fama-French 3, CAR windows, unit test against a hand computation | §7.5 | none |
| Q3 | S1 | `natural_experiments.csv`: at least 15 candidates with dates and source URLs (EDGAR 8-K 5.02, Wikipedia lists) | §11.2 | none |
| Q4 | S2 | `events.py`: z-scores, detection, clustering, filters, on fixtures, then on the real panel | §7.1–7.4 | fixtures; real: D11 |
| Q5 | S2 | `events` table written, with natural experiments flagged as reserved | §7.7 | Q4 |
| Q6 | S2 | `splits.py`: chronological train / validation / test with the 10-session gap | §9.5 | Q5 |
| Q7 | S2 | Notebook `02_event_study`: mean CAR by sign and group, tests, CAR curve from J-10 to J+10 | §7.6 | Q5 |
| Q8 | S3 | Placebo framework (random dates, unrelated people, wrong company, time reversal) | §11.3 | Q5 |
| Q9 | S3 | Robustness runner: re-runs the pipeline over the parameter variants of §11.4 | §11.4 | Q5 |
| Q10 | S4 | Natural experiment validation: ex ante score against real reaction | §11.2 | M10 merged |
| Q11 | S4 | Tests of H2 to H5 | §11.5 | M10 merged |
| Q12 | S4 | Nvidia 2026 case study (net-of-market CAR, competing news) | §11.6 | M10 merged |

### 7.3 Session ml

| # | Week | Task | Dossier | Needs |
| --- | --- | --- | --- | --- |
| M1 | S1 | Streamlit multipage skeleton (all pages of §12.2) reading fixture-shaped files from `app/data/` | §12 | none |
| M2 | S1 | `features.py`: the 5 feature families, no look-ahead, on fixture events | §9.2 | fixtures |
| M3 | S1 | Baselines B0–B2 and the time-series CV harness (expanding window, gap) | §9.3, §9.5 | fixtures |
| M4 | S2 | Metrics module: MAE, RMSE, out-of-sample R², Spearman, PR-AUC, bootstrap confidence intervals | §9.7 | none |
| M5 | S2 | `tone.py`: FinBERT inference on headlines, batched, cached | §8.3 | fixtures; real: D12 |
| M6 | S2 | README draft: pitch, method in 5 lines, install, placeholder for the app link | §12.6 | none |
| M7 | S3 | Lasso / Elastic Net, Random Forest, gradient boosting, tuned with the CV harness | §9.4 | Q5, Q6 merged |
| M8 | S3 | Mixed-effects model (statsmodels) with person-level slopes on negative and positive tone | §9.4, §10.2 | Q5 merged |
| M9 | S3 | Feature selection suite: filter, wrapper, embedded, stability, all inside the CV | §9.6 | M7 |
| M10 | S3 | Scores: methods A and B, agreement check, λ calibration, V, 0–10 scale, shrinkage, smoothing, succession resets | §10 | M7, M8 |
| M11 | S3 | SHAP and analysis of the 20 largest errors | §9.8, §9.9 | M7 |
| M12 | S4 | Final test-set evaluation, run **once**, only after the user says go in `coordination/inbox/ml/` | §9.5 | M10 |
| M13 | S4 | App on real scores, `export-app`, honest-display rules, deployment notes | §12 | M10 |

---

## 8. Git workflow inside a session

```powershell
git fetch origin
git switch -c quant/events origin/main      # new task: branch from fresh main
# ... work, small commits with clear English messages ...
ruff check .
pytest tests/quant
git push -u origin quant/events
gh pr create --fill --label quant
```

- One task per PR, ideally under ~400 changed lines.
- After the user merges any PR: `git fetch origin`, then `git rebase origin/main` on your current branch.
- PR title: `[data] Add Wikipedia pageviews collector`. The PR body follows the template.
- Never `git push --force`. If a rebase rewrote your already-pushed branch, use
  `git push --force-with-lease` **on your own branch only**, and say so in the PR.

**Conflict playbook:**

| Situation | What to do |
| --- | --- |
| Rebase conflict in a file you own | Resolve it, run the tests, continue |
| Rebase conflict in a file you do not own | `git rebase --abort`, write an inbox note to the owner and to `user`, update your status as blocked, move to another task |
| You need a new column or a schema change | Contract procedure (§6) |
| You need a new dependency | Add it to your own `requirements/<area>.txt`, pinned, in your PR |
| Two sessions need the same helper | The owner of the closest module writes it; the other asks through the inbox. Never copy-paste a frozen file |

---

## 9. Coordination files

**Status:** `coordination/status/<session>.md`, rewritten by its owner at the end of every task.

```markdown
# Status: quant
Updated: 2026-12-01 18:40
Current: Q4 events detection (branch quant/events), tests passing on fixtures
Done: Q1 (PR merged), Q2 (PR merged), Q3 (PR open)
Blocked: real-data run of Q4 waits for D11 (panel)
Needs from others: none
Next: Q5
```

**Inbox:** a request is a **new** file `coordination/inbox/<to>/<YYYY-MM-DD>-<from>-<slug>.md`.

```markdown
From: ml
To: quant
Need: add column `car_0_3` to `events`
Why: robustness window asked in §11.4
Urgency: low (not blocking, M7 runs without it)
```

The recipient handles it, then moves the file to `coordination/inbox/<to>/done/` in its next PR.
`coordination/inbox/user/` holds everything that needs the user: manual checks, decisions
outside the dossier, contract requests, the go for the final test-set run.

---

## 10. Merge order and weekly integration (user)

**Daily merge order:** contract PRs first, then `data`, then `quant`, then `ml`. CI must be green.

**End of each week**, in `key-person-radar` on `main`:

1. Merge every ready PR in the order above.
2. Run the full pipeline on real data (`python -m kpr.cli` steps in order) and the full test suite.
3. Check the week's gate (dossier §14): S1 sources usable, S2 at least 1,500 usable shocks,
   S3 plausible scores, S4 app online.
4. Tag: `git tag v0.<week>-S<week>` then `git push origin --tags`.
5. Read `coordination/inbox/user/` and answer each file.

---

## 11. Weekly dependency picture

| Week | data | quant | ml |
| --- | --- | --- | --- |
| S1 | D1–D7, all independent of the others | Q1–Q3 on fixtures | M1–M3 on fixtures |
| S2 | D8–D11 → panel merged mid-week | Q4–Q7, switch to real panel once D11 is merged | M4–M6 |
| S3 | D12 headlines once Q5 is merged | Q8–Q9 | M7–M11 on real events once Q5 and Q6 are merged |
| S4 | D13 | Q10–Q12 once M10 is merged | M12–M13 |

The critical path runs D11 → Q5 → M7/M8 → M10 → Q10–Q12. If one of these PRs is open,
reviewing and merging it comes first.
