"""Ex ante score vs real reaction on reserved natural experiments.

Owner: quant session. Task Q10 in PARALLEL_SESSIONS.md §7. Dossier §11.2.

The candidate list lives in `config/natural_experiments.csv` (task Q3). `load_candidates`
serves events.py (Q5), which flags those events as reserved.

Validation protocol (Q10, §11.2):

1. The events and their 10-day neighbours are out of training (events.py, split "reserved").
2. Ex ante score: the person's score at the end of the year before the event.
3. Real reaction: CAR[0, 2] (and [0, 5]) from the market model, J0 on the exchange calendar.
4. Spearman rank correlation between the ex ante score and |CAR|, with a bootstrap interval.
5. Same with a naive score (mean daily Wikipedia views of the person the year before), and
   the paired bootstrap interval of the difference: the model must beat it.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from kpr.utils.logging import get_logger

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
CANDIDATES_PATH = REPO_ROOT / "config" / "natural_experiments.csv"

COLUMNS = ("event_id", "person_name", "company_id", "date", "kind", "wanted_by_company",
           "source_url", "notes")
KINDS = (
    "arrest",                # arrest or criminal charge of the person
    "death",                 # death in office
    "health",                # illness or emergency treatment disclosed
    "legal_action",          # regulator or court action against the person
    "resignation_personal",  # resignation for a personal reason (conduct, private life)
    "dismissal",             # sudden removal by the board
    "surprise_departure",    # sudden departure not explained by conduct or health
    "surprise_appointment",  # unexpected arrival of a new leader
)
# Days around a natural experiment kept out of training and calibration (dossier §11.2).
EXCLUSION_DAYS = 10


def load_candidates(path: str | Path = CANDIDATES_PATH) -> pd.DataFrame:
    """Read and check the candidate list; `date` is the UTC day the news broke."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(df.columns) != COLUMNS:
        raise ValueError(f"{path}: columns must be {COLUMNS}, got {tuple(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    bad_kind = sorted(set(df["kind"]) - set(KINDS))
    if bad_kind:
        raise ValueError(f"{path}: unknown kind {bad_kind}; allowed: {KINDS}")
    flags = df["wanted_by_company"].str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError(f"{path}: wanted_by_company must be true or false")
    df["wanted_by_company"] = flags.eq("true")
    if df["event_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate event_id")
    return df


# ---- validation (Q10) ------------------------------------------------------------------------

SCORE_COLUMNS = ("v_pct", "s_minus", "s_plus")  # ex ante scores tested, main one first
N_BOOT = 2000
SEED = 2026
# Letters that NFKD does not split into a base letter plus an accent.
_NO_DECOMPOSITION = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "ß": "ss",
                                   "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "œ": "oe"})


def slug(name: str) -> str:
    """'Lars Fruergaard Jørgensen' -> 'lars_fruergaard_jorgensen' (person_id convention)."""
    text = unicodedata.normalize("NFKD", name.translate(_NO_DECOMPOSITION))
    text = text.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def load_person_ids(path: Path | None = None) -> dict[str, str]:
    """Lower-case name or alias -> person_id, from config/persons.yaml (data session)."""
    path = path or REPO_ROOT / "config" / "persons.yaml"
    with open(path, encoding="utf-8") as f:
        persons = (yaml.safe_load(f) or {}).get("persons") or []
    out: dict[str, str] = {}
    for p in persons:
        pid = p["person_id"]
        out[pid] = pid
        for alias in p.get("aliases") or []:
            out[slug(alias)] = pid
    return out


def attach_person_ids(candidates: pd.DataFrame, known: Mapping[str, str]) -> pd.DataFrame:
    """Add person_id (kept if already there): persons.yaml match on the name slug."""
    out = candidates.copy()
    if "person_id" not in out.columns:
        slugs = out["person_name"].map(slug)
        out["person_id"] = slugs.map(lambda s: known.get(s, s))
    return out


def candidate_cars(candidates: pd.DataFrame, prices: pd.DataFrame, market: pd.DataFrame,
                   universe=None, params: Mapping | None = None) -> pd.DataFrame:
    """J0 and real CAR[0, 2] and CAR[0, 5] of each candidate (market model, §7.5)."""
    from kpr.build import events as E
    from kpr.build import returns as R

    params = params or E.load_params()
    markets = E.Markets(prices, market, universe)
    known = set(prices["company_id"])
    ev = candidates[candidates["company_id"].isin(known)].copy()
    missing = sorted(set(candidates["company_id"]) - known)
    if missing:
        log.warning("no prices for %s: those candidates are skipped", missing)
    ev["j0"] = pd.to_datetime([markets.j0(c, d) for c, d in zip(ev["company_id"], ev["date"],
                                                                 strict=True)])
    mapping = {c: markets.index_of(c) for c in ev["company_id"].unique()}
    ok = ev["j0"].notna()
    cars = R.compute_cars(ev.loc[ok, ["event_id", "company_id", "j0"]], prices, market, mapping,
                          windows={"car_0_2": (0, 2), "car_0_5": (0, 5)},
                          estimation_window=tuple(params["returns"]["estimation_window"]),
                          sessions_for=markets.sessions)
    return ev.merge(cars[["event_id", "car_0_2", "car_0_5", "n_est"]], on="event_id",
                    how="left")


def ex_ante_scores(candidates: pd.DataFrame, scores: pd.DataFrame,
                   columns: tuple[str, ...] = SCORE_COLUMNS) -> pd.DataFrame:
    """Score of each person at the end of the year before the event (§11.2, step 2).

    The row of the same company is used when it exists; otherwise the person's score at
    another company that year (e.g. Brian Niccol at Chipotle before joining Starbucks).
    `score_source` says which: "same_company", "other_company" or "none".
    """
    prev = scores.assign(year=scores["year"] + 1)  # end of year y-1 -> events of year y
    rows = []
    for c in candidates.itertuples(index=False):
        year = pd.Timestamp(c.date).year
        mine = prev[(prev["person_id"] == c.person_id) & (prev["year"] == year)]
        same = mine[mine["company_id"] == c.company_id]
        pick, source = (same, "same_company") if len(same) else (mine, "other_company")
        if pick.empty:
            source = "none"
        row = {"event_id": c.event_id, "score_source": source}
        row.update({col: pick[col].mean() if len(pick) else np.nan for col in columns})
        rows.append(row)
    return pd.DataFrame(rows, columns=["event_id", "score_source", *columns])


def naive_score(candidates: pd.DataFrame, wiki_views: pd.DataFrame) -> pd.Series:
    """Mean daily Wikipedia views of the person in the year before the event (§11.2, step 5)."""
    out = {}
    for c in candidates.itertuples(index=False):
        end = pd.Timestamp(c.date)
        page = wiki_views[(wiki_views["page_id"] == c.person_id)
                          & (wiki_views["date"] >= end - pd.DateOffset(years=1))
                          & (wiki_views["date"] < end)]
        out[c.event_id] = page["views_total"].mean() if len(page) else np.nan
    return pd.Series(out, name="naive_views", dtype=float)


def bootstrap_spearman(x: pd.Series, y: pd.Series, n_boot: int = N_BOOT,
                       seed: int = SEED) -> tuple[float, float, float]:
    """Spearman rho of (x, y) with a 95 % percentile bootstrap interval."""
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 4:
        return np.nan, np.nan, np.nan
    a, b = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    rho = stats.spearmanr(a, b).statistic
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(len(a), size=len(a))
        if np.ptp(a[i]) > 0 and np.ptp(b[i]) > 0:
            boots.append(stats.spearmanr(a[i], b[i]).statistic)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(rho), float(lo), float(hi)


def bootstrap_rho_difference(score: pd.Series, naive: pd.Series, y: pd.Series,
                             n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """rho(score, y) - rho(naive, y) with a paired 95 % bootstrap interval."""
    df = pd.concat([score, naive, y], axis=1).dropna()
    if len(df) < 4:
        return np.nan, np.nan, np.nan
    s, v, t = (df.iloc[:, k].to_numpy() for k in range(3))

    def diff(i):
        return stats.spearmanr(s[i], t[i]).statistic - stats.spearmanr(v[i], t[i]).statistic

    full = diff(np.arange(len(df)))
    rng = np.random.default_rng(seed)
    boots = [d for d in (diff(rng.integers(len(df), size=len(df))) for _ in range(n_boot))
             if np.isfinite(d)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(full), float(lo), float(hi)


def validate_scores(candidates: pd.DataFrame, prices: pd.DataFrame, market: pd.DataFrame,
                    scores: pd.DataFrame, wiki_views: pd.DataFrame, *, universe=None,
                    person_ids: Mapping[str, str] | None = None,
                    target: str = "car_0_2") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-candidate table and summary of the ex ante validation (§11.2, steps 2-5).

    The target is the magnitude |CAR|: an influence score predicts that the price moves a
    lot, not in which direction.
    """
    cands = attach_person_ids(candidates, person_ids if person_ids is not None
                              else load_person_ids())
    table = candidate_cars(cands, prices, market, universe)
    table = table.merge(ex_ante_scores(table, scores), on="event_id", how="left")
    table["naive_views"] = table["event_id"].map(naive_score(table, wiki_views))
    table["abs_car"] = table[target].abs()

    rows = []
    for col in (*SCORE_COLUMNS, "naive_views"):
        rho, lo, hi = bootstrap_spearman(table[col], table["abs_car"])
        row = {"score": col, "n": int(table[[col, "abs_car"]].dropna().shape[0]),
               "spearman": rho, "lo": lo, "hi": hi}
        if col != "naive_views":
            d, dlo, dhi = bootstrap_rho_difference(table[col], table["naive_views"],
                                                   table["abs_car"])
            row.update(vs_naive=d, vs_naive_lo=dlo, vs_naive_hi=dhi)
        rows.append(row)
    return table, pd.DataFrame(rows).set_index("score")


def main(argv: list[str] | None = None) -> int:
    from kpr.utils.io import data_dir, read_table, use_fixtures

    parser = argparse.ArgumentParser(prog="python -m kpr.validation.natural_experiments",
                                     description="Ex ante score vs real reaction (§11.2).")
    parser.add_argument("--target", choices=["car_0_2", "car_0_5"], default="car_0_2")
    parser.add_argument("--out-dir", type=Path, help="default: processed/validation/")
    args = parser.parse_args(argv)

    table, summary = validate_scores(
        load_candidates(), read_table("prices"), read_table("market"), read_table("scores"),
        read_table("wiki_views"), target=args.target)
    log.info("candidates with a CAR: %d, with an ex ante score: %d",
             int(table[args.target].notna().sum()), int((table["score_source"] != "none").sum()))
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(summary.to_string())

    out_dir = args.out_dir
    if out_dir is None and not use_fixtures():
        out_dir = data_dir() / "processed" / "validation"
    if out_dir is None:
        log.info("fixtures: nothing written (pass --out-dir)")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "natural_experiments.csv", index=False)
    summary.to_csv(out_dir / "natural_experiments_summary.csv")
    log.info("wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
