"""Earnings dates (8-K item 2.02) and officer changes (8-K item 5.02) from SEC EDGAR.

Owner: data session. Task D6 in PARALLEL_SESSIONS.md §7. Dossier §5.6.

- Source: the EDGAR submissions API (data.sec.gov/submissions/CIK##########.json and its
  older pages), one JSON per filer, cached in raw/edgar/. Companies are matched through the
  `sec_ciks` of config/universe.yaml (several CIKs when the filer changed: Google Inc. then
  Alphabet, TWDC then Walt Disney Co, BlackRock then the 2024 holding).
- Earnings date = `reportDate` of each 8-K whose items include 2.02 (the date of the event),
  else the filing date. Every 2.02 filing is kept: results, preliminary results, guidance
  revisions, Tesla's delivery reports; all are company information releases to filter out.
  In a year with fewer than 3 such dates (Berkshire, Strategy publish results with the 10-Q),
  the 10-Q / 10-K filing dates are added (`source` = "edgar_10q_10k"; otherwise
  "edgar_8k_202"). `quarter` is the calendar quarter of the period most likely reported
  (date minus 45 days).
- Foreign private issuers (Shopify, Spotify, BioNTech) file 6-K without items: not covered
  here, they go to the manual template of task D9 with the European companies.
- Item 5.02 filings (officer departures and appointments) are saved to
  raw/edgar/officer_changes.csv: a reliable list for natural experiments (dossier §5.6).
- SEC fair-access rules: a User-Agent with a contact (HTTP_CONTACT) and at most 10 requests per
  second (we stay well below).

Usage: python -m kpr.collect.edgar
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from kpr.collect.common import (
    append_collection_log,
    cached_get_json,
    http_session,
    load_params,
    load_universe,
)
from kpr.utils.io import raw_dir, write_table
from kpr.utils.logging import get_logger

log = get_logger(__name__)

BASE = "https://data.sec.gov/submissions"
SOURCE = "edgar_8k_202"
FALLBACK_SOURCE = "edgar_10q_10k"
FORMS = ("8-K",)
PERIODIC = ("10-Q", "10-K")
MIN_PER_YEAR = 3      # fewer 8-K 2.02 dates in a year -> add periodic report dates
NEAR_DAYS = 10        # a periodic report this close to an 8-K 2.02 date is the same release


def _filings_frame(block: dict) -> pd.DataFrame:
    cols = ["accessionNumber", "filingDate", "reportDate", "form", "items"]
    return pd.DataFrame({c: block.get(c, []) for c in cols})


def fetch_filings(session, cik: int) -> pd.DataFrame:
    """Every filing of one CIK (recent block plus the older pages)."""
    key = f"CIK{cik:010d}"
    _s, sub = cached_get_json(session, "edgar", key, f"{BASE}/{key}.json", pause=0.15)
    frames = [_filings_frame(sub.get("filings", {}).get("recent", {}))]
    for f in sub.get("filings", {}).get("files", []):
        name = f["name"]
        _s, older = cached_get_json(session, "edgar", name.removesuffix(".json"),
                                    f"{BASE}/{name}", pause=0.15)
        frames.append(_filings_frame(older))
    df = pd.concat(frames, ignore_index=True)
    df["cik"] = cik
    return df


def calendar_quarter(dates: pd.Series) -> pd.Series:
    """Calendar quarter of the period most likely reported by a release on `dates`."""
    shifted = dates - pd.DateOffset(days=45)
    return "Q" + shifted.dt.quarter.astype(str)


def select_items(filings: pd.DataFrame, item: str) -> pd.DataFrame:
    """8-K filings whose item list contains `item`, with the event date."""
    f = filings[filings["form"].isin(FORMS)].copy()
    f = f[f["items"].fillna("").str.split(",").apply(lambda xs: item in [x.strip() for x in xs])]
    report = pd.to_datetime(f["reportDate"].replace("", None), errors="coerce")
    f["date"] = report.fillna(pd.to_datetime(f["filingDate"]))
    return f


def earnings_from_filings(company_id: str, filings: pd.DataFrame, start: str,
                          end: str) -> pd.DataFrame:
    """8-K 2.02 dates; in years with fewer than MIN_PER_YEAR of them (Berkshire and Strategy
    publish results with the 10-Q), the 10-Q / 10-K filing dates are added."""
    f = select_items(filings, "2.02")
    f = f[(f["date"] >= start) & (f["date"] <= end)]
    main = pd.DataFrame({"date": f["date"].drop_duplicates().sort_values().values})
    main["source"] = SOURCE

    per = filings[filings["form"].isin(PERIODIC)]
    per = pd.DataFrame({"date": pd.to_datetime(per["filingDate"]).drop_duplicates().values})
    per = per[(per["date"] >= start) & (per["date"] <= end)]
    counts = main["date"].dt.year.value_counts()
    thin_years = [y for y in per["date"].dt.year.unique() if counts.get(y, 0) < MIN_PER_YEAR]
    per = per[per["date"].dt.year.isin(thin_years)]
    if len(main) and len(per):
        gaps = per["date"].apply(lambda d: (main["date"] - d).abs().min())
        per = per[gaps > np.timedelta64(NEAR_DAYS, "D")]
    per["source"] = FALLBACK_SOURCE

    out = pd.concat([main, per], ignore_index=True).sort_values("date").reset_index(drop=True)
    out.insert(0, "company_id", company_id)
    out["quarter"] = calendar_quarter(out["date"])
    return out[["company_id", "date", "quarter", "source"]]


def build(companies: list[dict], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    session = http_session()
    earnings, officers = [], []
    for c in companies:
        ciks = c.get("sec_ciks") or []
        if not ciks:
            continue
        filings = pd.concat([fetch_filings(session, cik) for cik in ciks], ignore_index=True)
        earnings.append(earnings_from_filings(c["company_id"], filings, start, end))
        o = select_items(filings, "5.02")
        o = o[(o["date"] >= start) & (o["date"] <= end)]
        officers.append(o.assign(company_id=c["company_id"])[
            ["company_id", "cik", "date", "filingDate", "accessionNumber", "items"]])
    return (pd.concat(earnings, ignore_index=True),
            pd.concat(officers, ignore_index=True).sort_values(["company_id", "date"]))


def check_counts(earnings: pd.DataFrame) -> pd.Series:
    """Releases per company and year, from each company's first release on (dossier §6.4
    expects about 4 per year in the US)."""
    e = earnings.assign(year=earnings["date"].dt.year)
    counts = e.groupby(["company_id", "year"]).size()
    first = e.groupby("company_id")["year"].min()
    full = [(c, y) for c in first.index for y in range(first[c], e["year"].max() + 1)]
    return counts.reindex(pd.MultiIndex.from_tuples(full, names=["company_id", "year"]),
                          fill_value=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kpr.collect.edgar")
    parser.parse_args(argv)
    period = load_params()["period"]
    companies = load_universe()
    earnings, officers = build(companies, period["start"], period["end"])
    write_table(earnings, "earnings")
    officers.to_csv(raw_dir("edgar") / "officer_changes.csv", index=False)
    counts = check_counts(earnings)
    odd = counts[(counts < 3) | (counts > 6)]
    append_collection_log("edgar", earnings=len(earnings), companies=earnings["company_id"]
                          .nunique(), officer_changes=len(officers), odd_years=len(odd))
    log.info("earnings: %d dates for %d companies; officer changes (5.02): %d",
             len(earnings), earnings["company_id"].nunique(), len(officers))
    if len(odd):
        log.warning("company-years with fewer than 3 or more than 6 releases "
                    "(late listings and partial years included):\n%s", odd.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
