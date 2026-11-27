"""Tests of hypotheses H2 to H5.

Owner: quant session. Task Q11 in PARALLEL_SESSIONS.md §7. Dossier §11.5 (hypotheses §2).

- H2 (asymmetry): bad news weighs more than good news. Distribution of S-/S+ per person,
  share above 1, one-sided sign test and Wilcoxon test on log(S-/S+).
- H3 (founders, media): founders still in office, and media figures, are more sensitive.
  One-sided Mann-Whitney tests of the scores: founders vs others, media vs control.
- H4 (stability): the score is stable while the person stays in office and changes at a
  succession. Spearman correlation of a person's score from one year to the next at the
  same company, and the jump at each CEO succession compared with ordinary yearly changes.
- H5 (Huang vs Pouyanné): direct comparison of S and V with their bootstrap intervals; if
  the intervals overlap, we say so.

Every test reads the frozen `scores` table (ml session, task M10). Persons are compared on
their mean over the years they are scored, so a person with ten years does not count ten
times. Scores: raw sensitivities (s_minus_raw, s_plus_raw) for H2, published scores
(s_minus, s_plus, v_pct) for H3 to H5.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from kpr.utils.logging import get_logger
from kpr.validation import event_study as ES

log = get_logger(__name__)

SCORE_COLUMNS = ("s_minus", "s_plus", "v_pct")
H5_PERSONS = ("jensen_huang", "patrick_pouyanne")


def per_person(scores: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Mean of `columns` per (person_id, company_id) over the scored years."""
    return scores.groupby(["person_id", "company_id"], as_index=False)[list(columns)].mean()


# ---- H2 ----------------------------------------------------------------------------------------

def h2_asymmetry(scores: pd.DataFrame) -> dict:
    """S-/S+ per person (raw sensitivities): share above 1 and one-sided tests."""
    p = per_person(scores, ["s_minus_raw", "s_plus_raw"])
    p = p[(p["s_minus_raw"] > 0) & (p["s_plus_raw"] > 0)]
    ratio = p["s_minus_raw"] / p["s_plus_raw"]
    out = {"n": len(ratio), "mean_ratio": ratio.mean(), "median_ratio": ratio.median(),
           "share_above_1": (ratio > 1).mean() if len(ratio) else np.nan,
           "p_sign": np.nan, "p_wilcoxon": np.nan}
    differs = ratio[ratio != 1]
    if len(differs):
        out["p_sign"] = float(stats.binomtest(int((differs > 1).sum()), len(differs),
                                              alternative="greater").pvalue)
    log_ratio = np.log(ratio)
    if len(ratio) >= 2 and (log_ratio != 0).any():
        out["p_wilcoxon"] = float(stats.wilcoxon(log_ratio, alternative="greater").pvalue)
    return out


# ---- H3 ----------------------------------------------------------------------------------------

def _one_sided_mw(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(stats.mannwhitneyu(a, b, alternative="greater").pvalue)


def h3_groups(scores: pd.DataFrame, mandates: pd.DataFrame,
              groups: Mapping[str, str]) -> pd.DataFrame:
    """Founders vs others and media vs control: means and one-sided Mann-Whitney p-values."""
    p = per_person(scores, SCORE_COLUMNS)
    founder = mandates.groupby(["person_id", "company_id"])["founder"].any()
    p["founder"] = [bool(founder.get((a, b), False))
                    for a, b in zip(p["person_id"], p["company_id"], strict=True)]
    p["group"] = p["company_id"].map(groups).fillna("unknown")
    rows = []
    for name, first, second in (("founders_vs_others", p["founder"], ~p["founder"]),
                                ("media_vs_control", p["group"] == "media",
                                 p["group"] == "control")):
        for col in SCORE_COLUMNS:
            a, b = p.loc[first, col], p.loc[second, col]
            rows.append({"comparison": name, "score": col, "n_1": int(a.notna().sum()),
                         "n_2": int(b.notna().sum()), "mean_1": a.mean(), "mean_2": b.mean(),
                         "p_mannwhitney": _one_sided_mw(a, b)})
    return pd.DataFrame(rows).set_index(["comparison", "score"])


# ---- H4 ----------------------------------------------------------------------------------------

def yearly_pairs(scores: pd.DataFrame, column: str) -> pd.DataFrame:
    """(score in y, score in y+1) for the same person and company."""
    s = scores[["person_id", "company_id", "year", column]]
    nxt = s.assign(year=s["year"] - 1).rename(columns={column: "next"})
    return s.merge(nxt, on=["person_id", "company_id", "year"]).dropna()


def successions(mandates: pd.DataFrame, role: str = "CEO") -> pd.DataFrame:
    """CEO changes: company_id, year (start year of the new CEO), outgoing, incoming."""
    ceo = mandates[mandates["role"] == role].sort_values(["company_id", "start"])
    rows = []
    for cid, g in ceo.groupby("company_id"):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            if g.loc[i, "person_id"] != g.loc[i - 1, "person_id"]:
                rows.append({"company_id": cid, "year": g.loc[i, "start"].year,
                             "outgoing": g.loc[i - 1, "person_id"],
                             "incoming": g.loc[i, "person_id"]})
    return pd.DataFrame(rows, columns=["company_id", "year", "outgoing", "incoming"])


def h4_stability(scores: pd.DataFrame, mandates: pd.DataFrame) -> pd.DataFrame:
    """Year-on-year Spearman within a mandate, and succession jumps vs ordinary changes.

    The jump compares the outgoing CEO's score in the year before the change with the
    incoming CEO's score in the year of the change (or the first year scored after it).
    """
    succ = successions(mandates)
    rows = []
    for col in SCORE_COLUMNS:
        pairs = yearly_pairs(scores, col)
        ordinary = (pairs["next"] - pairs[col]).abs()
        jumps = []
        for s in succ.itertuples(index=False):
            co = scores[scores["company_id"] == s.company_id]
            before = co[(co["person_id"] == s.outgoing) & (co["year"] < s.year)]
            after = co[(co["person_id"] == s.incoming) & (co["year"] >= s.year)]
            if len(before) and len(after):
                jumps.append(abs(after.sort_values("year")[col].iloc[0]
                                 - before.sort_values("year")[col].iloc[-1]))
        jumps = pd.Series(jumps, dtype=float)
        rho = (stats.spearmanr(pairs[col], pairs["next"]).statistic
               if len(pairs) > 2 and pairs[col].nunique() > 1 and pairs["next"].nunique() > 1
               else np.nan)
        rows.append({"score": col, "n_pairs": len(pairs), "spearman_yoy": rho,
                     "mean_abs_change": ordinary.mean(), "n_successions": len(jumps),
                     "mean_abs_jump": jumps.mean() if len(jumps) else np.nan,
                     "p_jump_larger": _one_sided_mw(jumps, ordinary)})
    return pd.DataFrame(rows).set_index("score")


# ---- H5 ----------------------------------------------------------------------------------------

INTERVALS = {"s_minus": ("s_minus_lo", "s_minus_hi"), "s_plus": ("s_plus_lo", "s_plus_hi"),
             "v_pct": ("v_lo", "v_hi")}


def h5_compare(scores: pd.DataFrame, persons: Sequence[str] = H5_PERSONS,
               year: int | None = None) -> pd.DataFrame:
    """S and V of two persons in the same year with their intervals; overlap verdicts."""
    a, b = persons
    sa, sb = scores[scores["person_id"] == a], scores[scores["person_id"] == b]
    common = sorted(set(sa["year"]) & set(sb["year"]))
    if not common:
        raise ValueError(f"no common scored year for {a} and {b}")
    year = year or common[-1]
    ra = sa[sa["year"] == year].iloc[0]
    rb = sb[sb["year"] == year].iloc[0]
    rows = []
    for col, (lo, hi) in INTERVALS.items():
        overlap = not (ra[lo] > rb[hi] or rb[lo] > ra[hi])
        if overlap:
            verdict = "intervals overlap: no conclusion"
        else:
            verdict = f"{a if ra[col] > rb[col] else b} higher (intervals disjoint)"
        rows.append({"score": col, "year": year, a: ra[col], f"{a}_lo": ra[lo],
                     f"{a}_hi": ra[hi], b: rb[col], f"{b}_lo": rb[lo], f"{b}_hi": rb[hi],
                     "overlap": overlap, "verdict": verdict})
    return pd.DataFrame(rows).set_index("score")


# ---- command -------------------------------------------------------------------------------

def run_all(scores: pd.DataFrame, mandates: pd.DataFrame, groups: Mapping[str, str],
            h5_persons: Sequence[str] = H5_PERSONS) -> dict[str, pd.DataFrame]:
    out = {
        "h2": pd.DataFrame([h2_asymmetry(scores)]),
        "h3": h3_groups(scores, mandates, groups),
        "h4": h4_stability(scores, mandates),
    }
    try:
        out["h5"] = h5_compare(scores, h5_persons)
    except ValueError as exc:
        log.warning("H5 skipped: %s", exc)
    return out


def main(argv: list[str] | None = None) -> int:
    from kpr.utils.io import data_dir, read_table, use_fixtures

    parser = argparse.ArgumentParser(prog="python -m kpr.validation.hypotheses",
                                     description="Tests of hypotheses H2 to H5 (§11.5).")
    parser.add_argument("--h5", nargs=2, default=list(H5_PERSONS), metavar="PERSON_ID",
                        help="the two persons compared in H5")
    parser.add_argument("--out-dir", type=Path, help="default: processed/validation/")
    args = parser.parse_args(argv)

    results = run_all(read_table("scores"), read_table("mandates"), ES.company_groups(),
                      args.h5)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        for name, table in results.items():
            print(f"\n== {name.upper()}\n{table.to_string()}")

    out_dir = args.out_dir
    if out_dir is None and not use_fixtures():
        out_dir = data_dir() / "processed" / "validation"
    if out_dir is None:
        log.info("fixtures: nothing written (pass --out-dir)")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        table.to_csv(out_dir / f"hypotheses_{name}.csv")
    log.info("wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
