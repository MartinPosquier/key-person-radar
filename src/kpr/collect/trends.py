"""Google Trends ratio person / company per year (pytrends), best effort, cached.

Owner: data session. Task D7 in PARALLEL_SESSIONS.md §7. Dossier §4.4 step 2, §4.5, §5.5.

- One request per person and window compares the person with the company of their first
  mandate, on the same 0-100 scale. Windows of at most 5 years give weekly data:
  2014-2018, 2019-2023 and 2024-2025 (dossier §5.5).
- Terms are Google Trends entities (topic ids) when a suggestion matches the name and looks
  like the right kind of entity ("Tim Cook - CEO of Apple" rather than the text "tim cook",
  dossier §4.5), else the plain text. The choice is saved in raw/google_trends/entities.json
  for review.
- ratio_person_company of a year = median over the weeks of that year of
  Trends(person) / Trends(company), weeks where the company is at 0 being skipped
  (dossier §4.4 step 2). The 0.10 selection threshold is applied downstream.
- pytrends is fragile: responses are cached one file per request, a run resumes where it
  stopped, and it stops cleanly after 3 consecutive HTTP 429 errors (dossier §5.5). Missing
  person-years stay missing in the table.

Usage: python -m kpr.collect.trends [--max-requests N]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd

from kpr.collect.common import append_collection_log, cache_key, load_persons, load_universe
from kpr.utils.io import raw_dir, write_table
from kpr.utils.logging import get_logger

log = get_logger(__name__)

SOURCE = "google_trends"
WINDOWS = [("2014-01-01", "2018-12-31"), ("2019-01-01", "2023-12-31"),
           ("2024-01-01", "2025-12-31")]
PERSON_HINTS = ("ceo", "executive", "business", "entrepreneur", "chairman", "chair", "founder",
                "president", "officer", "investor", "engineer", "banker", "manager", "director",
                "magnate", "billionaire", "philanthropist")
COMPANY_HINTS = ("company", "corporation", "manufacturer", "bank", "retail", "airline", "firm",
                 "conglomerate", "group", "service", "platform", "insur", "chain", "holding",
                 "developer", "publisher", "pharmaceutical", "brand", "software", "automaker",
                 "exchange", "corp")
PAUSE = 6.0
MAX_429 = 3


class TooManyRequests(RuntimeError):
    pass


def _client():
    from pytrends.request import TrendReq

    return TrendReq(hl="en-US", tz=0, timeout=(10, 30))


def choose_entity(suggestions: list[dict], name: str, hints: tuple[str, ...]) -> dict | None:
    """First suggestion whose title is the name and whose type contains a hint."""
    for s in suggestions:
        title = s.get("title", "").strip().lower()
        kind = s.get("type", "").lower()
        if title == name.strip().lower() and any(h in kind for h in hints):
            return {"mid": s["mid"], "title": s["title"], "type": s.get("type", "")}
    return None


def resolve_entities(client, persons: dict[str, str], companies: dict[str, str]) -> dict:
    """{key: {"term": mid or text, ...}} for persons and companies, cached in one JSON."""
    path = raw_dir(SOURCE) / "entities.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [(f"person:{k}", v, PERSON_HINTS) for k, v in persons.items()] + \
           [(f"company:{k}", v, COMPANY_HINTS) for k, v in companies.items()]
    out: dict = {}
    for key, name, hints in todo:
        if key in cache:
            continue
        try:
            sugg = client.suggestions(name)
        except Exception as exc:  # noqa: BLE001 - pytrends raises bare exceptions on 429
            # Rate-limited: use the plain text for this run, without caching, so that a later
            # run can still resolve the entity.
            log.warning("no Trends suggestions for %r (%s): plain text used", name, exc)
            out[key] = {"name": name, "term": name, "entity": None}
            continue
        chosen = choose_entity(sugg, name, hints)
        cache[key] = {"name": name, "term": chosen["mid"] if chosen else name,
                      "entity": chosen}
        path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
        time.sleep(1.0)
    return {**cache, **out}


def fetch_pair(client, person_term: str, company_term: str, start: str, end: str) -> pd.DataFrame:
    """Weekly interest of [person, company] on one scale; cached per request."""
    path = raw_dir(SOURCE) / f"{cache_key(person_term, company_term, start, end)}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    try:
        client.build_payload([person_term, company_term], timeframe=f"{start} {end}")
        df = client.interest_over_time()
    except Exception as exc:  # noqa: BLE001
        if "429" in str(exc):
            raise TooManyRequests(str(exc)) from exc
        raise
    if df.empty:
        df = pd.DataFrame(columns=[person_term, company_term])
    df = df.drop(columns=["isPartial"], errors="ignore")
    df = df.rename(columns={person_term: "person", company_term: "company"})
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)
    time.sleep(PAUSE)
    return df


def yearly_ratio(weekly: pd.DataFrame) -> pd.Series:
    """Median over the weeks of each year of person / company (company at 0 skipped)."""
    w = weekly[weekly["company"] > 0]
    if w.empty:
        return pd.Series(dtype=float)
    ratio = w["person"].astype(float) / w["company"].astype(float)
    return ratio.groupby(ratio.index.year).median()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kpr.collect.trends")
    parser.add_argument("--max-requests", type=int, default=10_000)
    args = parser.parse_args(argv)

    universe = {c["company_id"]: c for c in load_universe()}
    first = {}
    for p in sorted(load_persons(), key=lambda p: str(p["start"])):
        first.setdefault(p["person_id"], p)
    names = {pid: (p.get("aliases") or [pid.replace("_", " ")])[0].title()
             for pid, p in first.items()}
    company_names = {cid: c["trends_entity"] for cid, c in universe.items()}

    client = _client()
    rows, done, errors_429 = [], 0, 0
    try:
        entities = resolve_entities(client, names, company_names)
        for pid, p in first.items():
            pterm = entities[f"person:{pid}"]["term"]
            cterm = entities[f"company:{p['company_id']}"]["term"]
            for start, end in WINDOWS:
                if done >= args.max_requests:
                    break
                try:
                    weekly = fetch_pair(client, pterm, cterm, start, end)
                    errors_429 = 0
                except TooManyRequests as exc:
                    errors_429 += 1
                    log.warning("429 from Google Trends (%d in a row): %s", errors_429, exc)
                    if errors_429 >= MAX_429:
                        raise
                    time.sleep(60 * errors_429)
                    continue
                done += 1
                for year, r in yearly_ratio(weekly).items():
                    if not int(start[:4]) <= year <= int(end[:4]):
                        continue  # a week starting late December belongs to the window before
                    rows.append({"person_id": pid, "year": int(year), "ratio": float(r)})
    except TooManyRequests:
        log.error("stopped after %d consecutive 429 errors; rerun later to resume", MAX_429)

    if not rows:
        log.error("no Trends data collected")
        return 1
    df = (pd.DataFrame(rows).groupby(["person_id", "year"], as_index=False)["ratio"].median()
          .rename(columns={"ratio": "ratio_person_company"}))
    write_table(df, "trends_ratio")
    append_collection_log(SOURCE, requests=done, rows=len(df), persons=df["person_id"].nunique())
    log.info("trends_ratio: %d person-years for %d persons", len(df), df["person_id"].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
