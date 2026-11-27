"""Fetch CEOs, founders, chairs and mandate dates from Wikidata.

Owner: data session. Task D2 in PARALLEL_SESSIONS.md §7. Dossier §4.4, §5.6.

Two steps, both cached:

1. `python -m kpr.collect.wikidata` reads the statements of every company of the universe
   (found through its English Wikipedia page) and writes every candidate mandate
   (CEO P169, chair P488, founder P112, with start P580 / end P582 qualifiers and their
   precision) to raw/wikidata/candidates.csv. It uses the wbgetentities API: the SPARQL
   service timed out on these queries. scripts/data_draft_persons.py turns the candidates
   into a draft config/persons.yaml, which the team checks by hand.
2. `python -m kpr.collect.wikidata --mandates` writes the `mandates` table from the checked
   config/persons.yaml.
"""

from __future__ import annotations

import argparse

import pandas as pd

from kpr.collect.common import (
    append_collection_log,
    cache_key,
    cached_get_json,
    http_session,
    load_persons,
    load_universe,
)
from kpr.utils.io import raw_dir, write_table
from kpr.utils.logging import get_logger

log = get_logger(__name__)

API = "https://www.wikidata.org/w/api.php"
ROLE_PROPS = {"P169": "CEO", "P488": "chair", "P112": "founder"}
LOCAL_LANGS = ("fr", "de", "it", "nl", "da", "es")
HUMAN = "Q5"
# Wikidata time precision: 9 = year, 10 = month, 11 = day.
PRECISION = {9: "year", 10: "month", 11: "day"}


def _get_entities(session, ids: list[str], props: str, prefix: str, **extra) -> dict:
    """wbgetentities in batches of 50, cached; returns {qid: entity}."""
    out: dict = {}
    ids = sorted(set(ids))
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        params = {"action": "wbgetentities", "format": "json", "ids": "|".join(chunk),
                  "props": props, **extra}
        _status, payload = cached_get_json(session, "wikidata", prefix + cache_key(chunk, props),
                                           API, params, pause=1.0)
        out.update(payload.get("entities", {}))
    return out


def _time(qualifiers: dict, prop: str) -> tuple[str | None, str | None]:
    """(YYYY-MM-DD, precision) of the first time qualifier `prop`, or (None, None)."""
    for snak in qualifiers.get(prop, []):
        if snak.get("snaktype") != "value":
            continue
        value = snak["datavalue"]["value"]
        date = value["time"].lstrip("+")[:10].replace("-00", "-01")
        return date, PRECISION.get(value.get("precision"))
    return None, None


def parse_claims(entity: dict, company_id: str) -> list[dict]:
    """One row per CEO / chair / founder statement of a company entity."""
    rows = []
    for prop, role in ROLE_PROPS.items():
        for st in entity.get("claims", {}).get(prop, []):
            snak = st.get("mainsnak", {})
            if snak.get("snaktype") != "value" or st.get("rank") == "deprecated":
                continue
            quals = st.get("qualifiers", {})
            start, start_prec = _time(quals, "P580")
            end, end_prec = _time(quals, "P582")
            rows.append({
                "company_id": company_id, "role": role,
                "person_qid": snak["datavalue"]["value"]["id"],
                "start": start, "start_precision": start_prec,
                "end": end, "end_precision": end_prec,
            })
    return rows


def fetch_person_pages(session, qids: list[str]) -> pd.DataFrame:
    """Label, humanness, English and local Wikipedia titles and English aliases."""
    sites = ["enwiki"] + [f"{lang}wiki" for lang in LOCAL_LANGS]
    entities = _get_entities(session, qids, "sitelinks|aliases|labels|claims", "persons_",
                             languages="en", sitefilter="|".join(sites))
    rows = []
    for qid, ent in entities.items():
        links = ent.get("sitelinks", {})
        local = [f"{site[:-4]}:{v['title'].replace(' ', '_')}" for site, v in links.items()
                 if site != "enwiki"]
        label = ent.get("labels", {}).get("en", {}).get("value", "")
        aliases = [a["value"] for a in ent.get("aliases", {}).get("en", [])]
        kinds = [c["mainsnak"].get("datavalue", {}).get("value", {}).get("id")
                 for c in ent.get("claims", {}).get("P31", [])]
        rows.append({
            "person_qid": qid,
            "person_label": label,
            "is_human": HUMAN in kinds,
            "enwiki": links.get("enwiki", {}).get("title", "").replace(" ", "_") or None,
            "local_pages": "|".join(sorted(local)),
            "aliases": "|".join(dict.fromkeys(x.lower() for x in [label, *aliases] if x)),
        })
    return pd.DataFrame(rows, columns=["person_qid", "person_label", "is_human", "enwiki",
                                       "local_pages", "aliases"])


def resolve_company_qids(session, companies: list[dict]) -> dict[str, str]:
    """Map Wikidata QID -> company_id through the English Wikipedia title."""
    out: dict[str, str] = {}
    by_title = {c["wiki_pages"]["en"].replace("_", " "): c["company_id"] for c in companies}
    titles = sorted(by_title)
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        params = {"action": "wbgetentities", "format": "json", "sites": "enwiki",
                  "titles": "|".join(chunk), "props": "sitelinks", "sitefilter": "enwiki",
                  }
        _status, payload = cached_get_json(session, "wikidata", "companies_" + cache_key(chunk),
                                           API, params, pause=1.0)
        for qid, ent in payload.get("entities", {}).items():
            title = ent.get("sitelinks", {}).get("enwiki", {}).get("title")
            if qid.startswith("Q") and title in by_title:
                out[qid] = by_title[title]
    missing = set(by_title.values()) - set(out.values())
    if missing:
        log.warning("no Wikidata item found for %s", sorted(missing))
    return out


def fetch_candidates(companies: list[dict]) -> pd.DataFrame:
    session = http_session()
    qid_to_company = resolve_company_qids(session, companies)
    entities = _get_entities(session, list(qid_to_company), "claims", "claims_")
    rows = [r for qid, ent in entities.items() for r in parse_claims(ent, qid_to_company[qid])]
    df = pd.DataFrame(rows).drop_duplicates(["company_id", "role", "person_qid", "start"])
    pages = fetch_person_pages(session, df["person_qid"].unique().tolist())
    df = df.merge(pages, on="person_qid", how="left")
    df = df[df["is_human"].fillna(False).astype(bool)].drop(columns="is_human")
    df = df.sort_values(["company_id", "role", "start"], na_position="first")
    missing = [c["company_id"] for c in companies if c["company_id"] not in set(df["company_id"])]
    if missing:
        log.warning("no Wikidata mandate at all for %s", missing)
    return df.reset_index(drop=True)


def mandates_from_persons(persons: list[dict]) -> pd.DataFrame:
    """Build the `mandates` table (frozen schema) from config/persons.yaml."""
    rows = [{
        "person_id": p["person_id"],
        "company_id": p["company_id"],
        "role": p["role"],
        "start": pd.Timestamp(str(p["start"])),
        "end": pd.Timestamp(str(p["end"])) if p.get("end") else pd.NaT,
        "founder": bool(p.get("founder", False)),
        "birth_date": pd.Timestamp(str(p["birth_date"])) if p.get("birth_date") else pd.NaT,
    } for p in persons]
    df = pd.DataFrame(rows, columns=["person_id", "company_id", "role", "start", "end",
                                     "founder", "birth_date"])
    df["birth_date"] = pd.to_datetime(df["birth_date"])
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    return df.sort_values(["person_id", "company_id", "start"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kpr.collect.wikidata")
    parser.add_argument("--mandates", action="store_true",
                        help="write the mandates table from config/persons.yaml")
    args = parser.parse_args(argv)

    if args.mandates:
        df = mandates_from_persons(load_persons())
        path = write_table(df, "mandates")
        log.info("wrote %d mandates to %s", len(df), path)
        return 0

    df = fetch_candidates(load_universe())
    out = raw_dir("wikidata") / "candidates.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    append_collection_log("wikidata", rows=len(df), companies=df["company_id"].nunique())
    log.info("wrote %d candidate mandates for %d companies to %s",
             len(df), df["company_id"].nunique(), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
