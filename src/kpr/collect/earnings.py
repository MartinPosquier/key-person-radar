"""Merge earnings dates (EDGAR, yfinance, manual EU CSV) into the earnings table.

Owner: data session. Task D9 in PARALLEL_SESSIONS.md §7. Dossier §5.6, §6.4.

Europe has no central free source (dossier §5.6), and three US-listed foreign issuers
(Shopify, Spotify, BioNTech) file no 8-K. Their dates live in a CSV that the team completes
by hand: config/earnings_manual.csv, one row per release.

  company_id,date,quarter,source,checked_by,note
  MC,2019-01-29,Q4,yfinance,,             <- pre-filled, to confirm
  MC,2019-07-25,Q2,manual,alice,"H1 results, investor site"

- `python -m kpr.collect.earnings --template` (re)builds the CSV: it keeps every existing row,
  adds the release dates yfinance knows for the manual companies (`source = yfinance`), and
  adds one empty row per missing company-year so that the gaps are visible.
- `python -m kpr.collect.earnings` writes the `earnings` table: EDGAR dates
  (raw/edgar/earnings_edgar.parquet, task D6) plus every CSV row that has a date, and prints
  the company-years that look incomplete (dossier §6.4: about 4 releases a year in the US,
  2 to 4 in Europe).
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from kpr.collect.common import CONFIG_DIR, append_collection_log, load_params, load_universe
from kpr.utils.io import raw_dir, write_table
from kpr.utils.logging import get_logger

log = get_logger(__name__)

MANUAL_CSV = CONFIG_DIR / "earnings_manual.csv"
COLUMNS = ["company_id", "date", "quarter", "source", "checked_by", "note"]
EDGAR_FILE = "earnings_edgar.parquet"
EXPECTED = {"US": (3, 6), "EU": (2, 5)}  # plausible releases per full year


def manual_companies(companies: list[dict]) -> list[dict]:
    """Companies whose dates are not covered by EDGAR 8-K filings."""
    return [c for c in companies if not c.get("sec_ciks")]


def read_manual() -> pd.DataFrame:
    if not MANUAL_CSV.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(MANUAL_CSV, dtype=str, keep_default_na=False)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[COLUMNS]


def yfinance_dates(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Past release dates known to yfinance (local date of the exchange), cached."""
    path = raw_dir("yfinance") / f"earnings_{ticker.replace('.', '_')}.csv"
    if path.exists():
        raw = pd.read_csv(path)
    else:
        import yfinance as yf

        try:
            got = yf.Ticker(ticker).get_earnings_dates(limit=100)
        except Exception as exc:  # noqa: BLE001 - yfinance raises many kinds of errors
            log.warning("no yfinance earnings dates for %s: %s", ticker, exc)
            got = None
        raw = pd.DataFrame({"when": [] if got is None else got.index.astype(str)})
        raw.to_csv(path, index=False)
        time.sleep(0.5)
    when = pd.to_datetime(raw["when"], utc=True, errors="coerce").dropna()
    dates = when.dt.tz_convert("Europe/Paris").dt.tz_localize(None).dt.normalize()
    dates = dates[(dates >= start) & (dates <= end)].drop_duplicates().sort_values()
    return pd.DataFrame({"date": dates.values})


QUARTER_LAG_DAYS = 75


def calendar_quarter(dates: pd.Series) -> pd.Series:
    """Calendar quarter of the period most likely reported by a release on `dates`.

    Releases come 2 to 10 weeks after the quarter end: 75 days back lands in the reported
    quarter for both early US releases (mid-January -> Q4) and late European annual
    results (mid-March -> Q4).
    """
    return "Q" + (dates - pd.DateOffset(days=QUARTER_LAG_DAYS)).dt.quarter.astype(str)


def build_template(companies: list[dict], existing: pd.DataFrame, start: str,
                   end: str) -> pd.DataFrame:
    """Existing rows + yfinance dates not yet listed + one empty row per empty company-year."""
    rows = [existing[existing["date"] != ""]]
    for c in manual_companies(companies):
        cid = c["company_id"]
        have = set(existing.loc[existing["company_id"] == cid, "date"])
        yf_dates = yfinance_dates(c["ticker"], start, end)
        new = yf_dates[~yf_dates["date"].dt.strftime("%Y-%m-%d").isin(have)]
        if len(new):
            rows.append(pd.DataFrame({
                "company_id": cid, "date": new["date"].dt.strftime("%Y-%m-%d"),
                "quarter": calendar_quarter(new["date"]).values, "source": "yfinance",
                "checked_by": "", "note": "pre-filled from yfinance, to confirm"}))
    filled = pd.concat(rows, ignore_index=True)
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    gaps = []
    for c in manual_companies(companies):
        cid = c["company_id"]
        got = set(pd.to_datetime(filled.loc[filled["company_id"] == cid, "date"]).dt.year)
        for y in years:
            if y not in got:
                gaps.append({"company_id": cid, "date": "", "quarter": "", "source": "manual",
                             "checked_by": "", "note": f"{y}: add every release of this year"})
    out = pd.concat([filled, pd.DataFrame(gaps, columns=COLUMNS)], ignore_index=True)
    out["_year"] = out["date"].str[:4].where(out["date"] != "", out["note"].str[:4])
    out = out.sort_values(["company_id", "_year", "date"]).drop(columns="_year")
    return out[COLUMNS].reset_index(drop=True)


def merge(edgar: pd.DataFrame, manual: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """The `earnings` table: EDGAR rows plus manual rows that have a date."""
    m = manual[manual["date"] != ""].copy()
    m["date"] = pd.to_datetime(m["date"])
    m["quarter"] = m["quarter"].replace("", None)
    m["source"] = m["source"].replace("", "manual")
    parts = [edgar[["company_id", "date", "quarter", "source"]],
             m[["company_id", "date", "quarter", "source"]]]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    df = df.drop_duplicates(["company_id", "date"]).sort_values(["company_id", "date"])
    return df.reset_index(drop=True)


def incomplete_years(df: pd.DataFrame, companies: list[dict], end: str) -> pd.DataFrame:
    """Company-years outside the plausible range, from each company's first release on."""
    region = {c["company_id"]: c["region"] for c in companies}
    e = df.assign(year=df["date"].dt.year)
    counts = e.groupby(["company_id", "year"]).size()
    rows = []
    last = pd.Timestamp(end).year
    for cid in sorted(region):
        lo, hi = EXPECTED[region[cid]]
        first = e.loc[e["company_id"] == cid, "year"].min()
        if pd.isna(first):
            rows.append({"company_id": cid, "year": "all", "releases": 0})
            continue
        for y in range(int(first) + 1, last + 1):  # the first year can be partial
            n = int(counts.get((cid, y), 0))
            if n < lo or n > hi:
                rows.append({"company_id": cid, "year": y, "releases": n})
    return pd.DataFrame(rows, columns=["company_id", "year", "releases"])


def write_earnings() -> pd.DataFrame:
    period = load_params()["period"]
    companies = load_universe()
    path = raw_dir("edgar") / EDGAR_FILE
    edgar = pd.read_parquet(path) if path.exists() else pd.DataFrame(
        columns=["company_id", "date", "quarter", "source"])
    df = merge(edgar, read_manual(), period["start"], period["end"])
    write_table(df, "earnings")
    odd = incomplete_years(df, companies, period["end"])
    append_collection_log("earnings", rows=len(df), companies=df["company_id"].nunique(),
                          incomplete_company_years=len(odd))
    log.info("earnings: %d dates for %d companies (%s)", len(df), df["company_id"].nunique(),
             df["source"].value_counts().to_dict())
    if len(odd):
        log.warning("%d company-years look incomplete:\n%s", len(odd), odd.to_string(index=False))
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kpr.collect.earnings")
    parser.add_argument("--template", action="store_true",
                        help="(re)build config/earnings_manual.csv, keeping existing rows")
    args = parser.parse_args(argv)
    if args.template:
        period = load_params()["period"]
        out = build_template(load_universe(), read_manual(), period["start"], period["end"])
        out.to_csv(MANUAL_CSV, index=False, encoding="utf-8")
        filled = (out["date"] != "").sum()
        log.info("%s: %d rows, %d with a date, %d company-years to fill", MANUAL_CSV.name,
                 len(out), filled, len(out) - filled)
        return 0
    write_earnings()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
