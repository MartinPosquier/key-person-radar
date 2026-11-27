"""Load FNSPID headlines (fallback tone source).

Owner: data session. Task D8 in PARALLEL_SESSIONS.md §7. Dossier §5.7, §8.3.

FNSPID (Hugging Face dataset Zihan1004/FNSPID) gives financial news titles linked to US
tickers until 2023. We use `Stock_news/All_external.csv` (5.7 GB: date in UTC, title, ticker,
URL, publisher); the 23 GB file with full articles is not needed (dossier §8.3 works on
titles only).

  python -m kpr.collect.fnspid --download   # resumable download to raw/fnspid/
  python -m kpr.collect.fnspid --subset     # keep the rows that concern our universe

The subset keeps a title when its ticker is one of our US tickers or when it contains one of
the persons' aliases, and saves raw/fnspid/subset.parquet (person matching is done later,
per use: headlines for D12, daily counts if FNSPID replaces GDELT in D8).
Limit: FNSPID only covers US tickers, so European companies are matched through aliases in
titles only.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import requests

from kpr.collect.common import append_collection_log, load_persons, load_universe, user_agent
from kpr.utils.io import raw_dir
from kpr.utils.logging import get_logger

log = get_logger(__name__)

URL = ("https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/"
       "Stock_news/All_external.csv")
FILE = "All_external.csv"
COLUMNS = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher"]
CHUNK_ROWS = 500_000


def download(url: str = URL, attempts: int = 30) -> str:
    """Resumable download (HTTP Range) to raw/fnspid/All_external.csv, retried on cuts."""
    import time

    for i in range(attempts):
        try:
            return _download_once(url)
        except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError,
                requests.Timeout) as exc:
            log.warning("download cut (%s), resuming (%d / %d)", exc.__class__.__name__,
                        i + 1, attempts)
            time.sleep(5)
    raise RuntimeError("FNSPID download failed after repeated cuts; rerun to resume")


def _download_once(url: str) -> str:
    path = raw_dir("fnspid") / FILE
    part = path.with_suffix(".part")
    if path.exists():
        return str(path)
    done = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": user_agent()}
    if done:
        headers["Range"] = f"bytes={done}-"
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        if r.status_code == 416:  # already complete
            os.replace(part, path)
            return str(path)
        r.raise_for_status()
        mode = "ab" if done and r.status_code == 206 else "wb"
        total = int(r.headers.get("Content-Length", 0)) + (done if mode == "ab" else 0)
        written = done if mode == "ab" else 0
        with open(part, mode) as fh:
            for block in r.iter_content(chunk_size=8 << 20):
                fh.write(block)
                written += len(block)
                if written // (500 << 20) != (written - len(block)) // (500 << 20):
                    log.info("downloaded %.1f / %.1f GB", written / 1e9, total / 1e9)
    os.replace(part, path)
    append_collection_log("fnspid", file=FILE, bytes=written)
    return str(path)


def alias_pattern(aliases: list[str]) -> str:
    """Case-insensitive regex alternation of aliases, longest first, on word boundaries."""
    import re

    uniq = sorted({a.strip().lower() for a in aliases if a.strip()}, key=len, reverse=True)
    return r"\b(?:" + "|".join(re.escape(a) for a in uniq) + r")\b"


def subset(path: str, tickers: set[str], aliases: list[str]) -> pd.DataFrame:
    """Stream the CSV and keep rows of our tickers or whose title names one of our persons."""
    pattern = alias_pattern(aliases)
    kept = []
    n = 0
    reader = pd.read_csv(path, usecols=COLUMNS, chunksize=CHUNK_ROWS, dtype=str,
                         on_bad_lines="skip", encoding_errors="replace")
    for chunk in reader:
        n += len(chunk)
        title = chunk["Article_title"].fillna("")
        mask = chunk["Stock_symbol"].isin(tickers) | title.str.lower().str.contains(
            pattern, regex=True)
        kept.append(chunk[mask])
        log.info("%d rows read, %d kept so far", n, sum(len(k) for k in kept))
    df = pd.concat(kept, ignore_index=True)
    df["date_utc"] = pd.to_datetime(df["Date"].str.replace(" UTC", "", regex=False),
                                    errors="coerce", utc=True).dt.tz_localize(None)
    return df.drop(columns=["Date"]).rename(columns={
        "Article_title": "title", "Stock_symbol": "ticker", "Url": "url",
        "Publisher": "publisher"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kpr.collect.fnspid")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--subset", action="store_true")
    args = parser.parse_args(argv)
    if not (args.download or args.subset):
        parser.error("choose --download and/or --subset")
    path = str(raw_dir("fnspid") / FILE)
    if args.download:
        path = download()
        log.info("FNSPID file ready: %s", path)
    if args.subset:
        tickers = {c["ticker"] for c in load_universe() if c["region"] == "US"}
        aliases = [a for p in load_persons() for a in (p.get("aliases") or [])]
        df = subset(path, tickers, aliases)
        out = raw_dir("fnspid") / "subset.parquet"
        df.to_parquet(out, index=False)
        append_collection_log("fnspid", subset_rows=len(df))
        log.info("subset: %d titles -> %s", len(df), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
