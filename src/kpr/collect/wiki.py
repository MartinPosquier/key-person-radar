"""Download daily Wikipedia pageviews (English + local language), cached.

Owner: data session. Task D4 in PARALLEL_SESSIONS.md §7. Dossier §5.2, §4.5, §6.4.

- Pages: every person of config/persons.yaml (page_id = person_id) and every company of
  config/universe.yaml (page_id = "company_<company_id>"), in English and, when listed, in
  the local language (fr, de, ...).
- Agent `user`, access `all-access`, daily, 2015-07-01 (start of the API) to the end of the
  study period.
- Renamed pages: views made under an old title stay on that title, which is now a redirect.
  Each page's views are therefore the sum over the canonical title and its former titles
  (dossier §4.5: "suivre les redirections et les renommages de page"). Former titles are the
  English redirects tagged "R from move" (category "Redirects from moves"); other redirects
  (spelling variants, typos) carry little traffic and would multiply the requests by 15
  under Wikimedia's rate limits.
- A day without any data stays missing (never zero, dossier §6.4). `views_local` is 0 when the
  page has no local-language version, so that views_total = views_en + views_local.
- Raw responses are cached in raw/wikipedia_pageviews/ and raw/wikipedia_redirects/; an
  interrupted run resumes where it stopped.

Usage: python -m kpr.collect.wiki [--no-redirects] [--only PAGE_ID ...]
"""

from __future__ import annotations

import argparse
from urllib.parse import quote

import pandas as pd

from kpr.collect.common import (
    append_collection_log,
    cache_key,
    cached_get_json,
    http_session,
    load_params,
    load_persons,
    load_universe,
)
from kpr.utils.io import write_table
from kpr.utils.logging import get_logger

log = get_logger(__name__)

API_START = "2015-07-01"
REST = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"


def page_list() -> pd.DataFrame:
    """One row per (page_id, lang, title)."""
    rows = []
    seen: set[tuple[str, str]] = set()
    for p in load_persons():
        for lang, title in (p.get("wiki_pages") or {}).items():
            key = (p["person_id"], lang)
            if title and key not in seen:
                seen.add(key)
                rows.append({"page_id": p["person_id"], "lang": lang, "title": title})
    for c in load_universe():
        for lang, title in c["wiki_pages"].items():
            rows.append({"page_id": f"company_{c['company_id']}", "lang": lang, "title": title})
    return pd.DataFrame(rows, columns=["page_id", "lang", "title"])


def fetch_redirects(session, lang: str, titles: list[str]) -> dict[str, list[str]]:
    """{canonical title: [redirect titles]} for article-namespace redirects."""
    out: dict[str, list[str]] = {t: [] for t in titles}
    url = f"https://{lang}.wikipedia.org/w/api.php"
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        cont: dict = {}
        page = 0
        while True:
            params = {"action": "query", "format": "json", "formatversion": 2,
                      "titles": "|".join(t.replace("_", " ") for t in chunk),
                      "prop": "redirects", "rdnamespace": 0, "rdlimit": "max", "rdprop": "title",
                      **cont}
            _s, payload = cached_get_json(session, "wikipedia_redirects",
                                          f"{lang}_{cache_key(chunk, page)}", url, params,
                                          pause=1.0, retries=6)
            for p in payload.get("query", {}).get("pages", []):
                title = p["title"].replace(" ", "_")
                if title in out:
                    out[title].extend(r["title"].replace(" ", "_")
                                      for r in p.get("redirects", []))
            if "continue" not in payload:
                break
            cont = payload["continue"]
            page += 1
    return out


MOVED_CATEGORY = {"en": "Category:Redirects from moves"}


def keep_moved(session, lang: str, redirects: list[str]) -> list[str]:
    """Redirects that are former titles of the page (tagged as redirects from moves)."""
    category = MOVED_CATEGORY.get(lang)
    if not category or not redirects:
        return []
    url = f"https://{lang}.wikipedia.org/w/api.php"
    kept: list[str] = []
    for i in range(0, len(redirects), 50):
        chunk = redirects[i:i + 50]
        params = {"action": "query", "format": "json", "formatversion": 2,
                  "titles": "|".join(t.replace("_", " ") for t in chunk), "prop": "categories",
                  "clcategories": category, "cllimit": "max"}
        _s, payload = cached_get_json(session, "wikipedia_redirects",
                                      f"moved_{lang}_{cache_key(chunk)}", url, params,
                                      pause=1.0, retries=6)
        for p in payload.get("query", {}).get("pages", []):
            if p.get("categories"):
                kept.append(p["title"].replace(" ", "_"))
    return kept


def fetch_views(session, lang: str, title: str, start: str, end: str) -> pd.Series:
    """Daily user views of one title; empty Series if the API has no data (404)."""
    url = (f"{REST}/{lang}.wikipedia/all-access/user/{quote(title, safe='')}/daily/"
           f"{start.replace('-', '')}00/{end.replace('-', '')}00")
    status, payload = cached_get_json(session, "wikipedia_pageviews",
                                      f"{lang}_{cache_key(title, start, end)}", url,
                                      pause=0.05, retries=6, ok_statuses=(200, 404))
    if status == 404 or not isinstance(payload, dict):
        return pd.Series(dtype=float)
    items = payload.get("items", [])
    if not items:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([it["timestamp"][:8] for it in items], format="%Y%m%d")
    return pd.Series([float(it["views"]) for it in items], index=idx).groupby(level=0).sum()


def page_views(session, lang: str, titles: list[str], start: str, end: str) -> pd.Series:
    """Sum of daily views over several titles; missing only when no title has data."""
    series = [s for s in (fetch_views(session, lang, t, start, end) for t in titles) if len(s)]
    if not series:
        return pd.Series(dtype=float)
    return pd.concat(series, axis=1).sum(axis=1, min_count=1)


def build_table(pages: pd.DataFrame, views: dict[tuple[str, str], pd.Series],
                start: str, end: str) -> pd.DataFrame:
    days = pd.date_range(start, end, freq="D", name="date")
    frames = []
    for page_id, g in pages.groupby("page_id", sort=False):
        langs = set(g["lang"])
        if "en" in langs:
            en = views.get((page_id, "en"), pd.Series(dtype=float)).reindex(days)
        else:
            en = pd.Series(0.0, index=days)  # no English page at all: local views only
        local_langs = sorted(langs - {"en"})
        if local_langs:
            parts = [views.get((page_id, lang), pd.Series(dtype=float)).reindex(days)
                     for lang in local_langs]
            local = pd.concat(parts, axis=1).sum(axis=1, min_count=1)
        else:
            local = pd.Series(0.0, index=days)
        total = en.fillna(0.0) + local.fillna(0.0)
        total[en.isna() & local.isna()] = float("nan")
        frames.append(pd.DataFrame({"page_id": page_id, "date": days, "views_en": en.values,
                                    "views_local": local.values, "views_total": total.values}))
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kpr.collect.wiki")
    parser.add_argument("--no-redirects", action="store_true",
                        help="only the canonical titles (misses renamed pages)")
    parser.add_argument("--only", nargs="*", help="limit to these page_ids (no table written)")
    args = parser.parse_args(argv)

    start, end = API_START, load_params()["period"]["end"]
    pages = page_list()
    if args.only:
        pages = pages[pages["page_id"].isin(args.only)]
    session = http_session()

    redirects: dict[tuple[str, str], list[str]] = {}
    if not args.no_redirects:
        for lang, g in pages.groupby("lang"):
            found = fetch_redirects(session, lang, sorted(set(g["title"])))
            all_redirects = sorted({r for rs in found.values() for r in rs})
            moved = set(keep_moved(session, lang, all_redirects))
            for t, rs in found.items():
                redirects[(lang, t)] = [r for r in rs if r in moved]
            log.info("%s: %d redirects, %d former titles kept", lang, len(all_redirects),
                     len(moved))

    views: dict[tuple[str, str], pd.Series] = {}
    n_titles = 0
    for i, r in enumerate(pages.itertuples(), 1):
        titles = [r.title, *redirects.get((r.lang, r.title), [])]
        n_titles += len(titles)
        views[(r.page_id, r.lang)] = page_views(session, r.lang, titles, start, end)
        if i % 25 == 0:
            log.info("%d / %d pages (%d titles so far)", i, len(pages), n_titles)

    table = build_table(pages, views, start, end)
    if args.only:
        log.info("%s", table.groupby("page_id")["views_total"].median())
        return 0
    write_table(table, "wiki_views")
    miss = table.groupby("page_id")["views_total"].apply(lambda s: s.isna().mean())
    append_collection_log("wikipedia_pageviews", pages=len(pages), titles=n_titles,
                          rows=len(table), pages_with_missing=int((miss > 0).sum()))
    log.info("wiki_views: %d rows, %d pages, %d titles incl. redirects; "
             "%d pages with missing days", len(table), table["page_id"].nunique(), n_titles,
             int((miss > 0).sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
