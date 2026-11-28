"""Check config/universe.yaml against yfinance and Wikipedia (task D1, dossier §4.2).

For every company: the ticker returns daily prices over the study period (first and last
date, number of sessions, currency), and each Wikipedia title exists (redirects resolved,
canonical title reported). Also checks the market index and VIX tickers.

Usage:  python scripts/data_check_universe.py [--out report.md]
Exit code 1 if any ticker or page fails.
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd
import yfinance as yf

from kpr.collect.common import http_session, load_params, load_universe


def check_ticker(ticker: str, start: str, end: str) -> dict:
    t = yf.Ticker(ticker)
    hist = t.history(start=start, end=end, auto_adjust=True, actions=False)
    out = {"ticker": ticker, "n": len(hist), "first": None, "last": None, "currency": None}
    if len(hist):
        out["first"] = hist.index.min().date().isoformat()
        out["last"] = hist.index.max().date().isoformat()
    try:
        out["currency"] = (t.history_metadata or {}).get("currency")
    except Exception:  # noqa: BLE001 - metadata is best effort
        pass
    return out


def check_wiki_pages(session, lang: str, titles: list[str]) -> dict[str, dict]:
    """Return {asked title: {"canonical": str|None, "redirected": bool}}."""
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {"action": "query", "format": "json", "redirects": 1, "formatversion": 2,
              "titles": "|".join(t.replace("_", " ") for t in titles)}
    data = session.get(url, params=params, timeout=30).json()["query"]
    norm = {n["from"]: n["to"] for n in data.get("normalized", [])}
    redir = {r["from"]: r["to"] for r in data.get("redirects", [])}
    pages = {p["title"]: p for p in data.get("pages", [])}
    out = {}
    for asked in titles:
        title = asked.replace("_", " ")
        title = norm.get(title, title)
        final = redir.get(title, title)
        page = pages.get(final, {})
        exists = bool(page) and not page.get("missing") and not page.get("invalid")
        out[asked] = {"canonical": final.replace(" ", "_") if exists else None,
                      "redirected": final != title}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="write the markdown report here")
    args = parser.parse_args(argv)

    params = load_params()
    start, end = params["period"]["start"], params["period"]["end"]
    end_excl = (pd.Timestamp(end) + pd.DateOffset(days=1)).date().isoformat()
    companies = load_universe()
    session = http_session()

    rows, failures = [], []
    for c in companies:
        r = check_ticker(c["ticker"], start, end_excl)
        time.sleep(0.3)
        if r["n"] == 0:
            failures.append(f"{c['company_id']}: no prices for {c['ticker']}")
        rows.append({"company_id": c["company_id"], **r})

    wiki_rows = []
    by_lang: dict[str, list[tuple[str, str]]] = {}
    for c in companies:
        for lang, title in c["wiki_pages"].items():
            by_lang.setdefault(lang, []).append((c["company_id"], title))
    for lang, items in by_lang.items():
        for i in range(0, len(items), 40):
            chunk = items[i:i + 40]
            res = check_wiki_pages(session, lang, [t for _, t in chunk])
            for cid, title in chunk:
                info = res[title]
                wiki_rows.append({"company_id": cid, "lang": lang, "title": title, **info})
                if info["canonical"] is None:
                    failures.append(f"{cid}: {lang}:{title} does not exist")
                elif info["canonical"] != title:
                    failures.append(f"{cid}: {lang}:{title} -> use {info['canonical']}")

    index_rows = [check_ticker(t, start, end_excl) for t in ("^GSPC", "^STOXX", "^VIX")]
    for r in index_rows:
        if r["n"] == 0:
            failures.append(f"index {r['ticker']}: no data")

    lines = [f"# Universe check ({time.strftime('%Y-%m-%d')})", "",
             f"Period {start} to {end}. {len(companies)} companies.", "",
             "## Tickers (yfinance)", "",
             "| company_id | ticker | sessions | first | last | currency |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows + [{"company_id": "(index)", **r} for r in index_rows]:
        lines.append(f"| {r['company_id']} | {r['ticker']} | {r['n']} | {r['first']} | "
                     f"{r['last']} | {r['currency']} |")
    lines += ["", "## Wikipedia pages", "", "| company_id | lang | title | canonical |",
              "| --- | --- | --- | --- |"]
    for w in wiki_rows:
        lines.append(f"| {w['company_id']} | {w['lang']} | {w['title']} | {w['canonical']} |")
    lines += ["", "## Problems", ""] + ([f"- {f}" for f in failures] or ["- none"])
    report = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report)
    print(report)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
