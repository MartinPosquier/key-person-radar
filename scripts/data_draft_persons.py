"""Draft config/persons.yaml from a seed list and Wikidata, plus a manual-check list (task D2).

Dossier §4.4 step 1 (broad list) and §5.6. The seed below holds every CEO in office during
2014-2025 for the 60 companies, plus the other candidates named in dossier §4.2 (founders,
chairs, designated successors). Dates in the seed are a first draft. For each mandate the
script:

- resolves the English Wikipedia title (canonical title, Wikidata QID, local-language page
  for European companies, English aliases for GDELT matching) and the birth date (P569);
- compares start and end with the Wikidata statements of raw/wikidata/candidates.csv
  (run `python -m kpr.collect.wikidata` first) and records the outcome in `source_check`;
- lists every mandate that needs a human check, and every Wikidata CEO of the period that
  the seed does not contain.

`end` is the last day in the role (inclusive), null if still in office. Co-CEO mandates are
flagged `co_ceo: true` so that the overlap check of dossier §6.7 accepts them.

Usage:  python scripts/data_draft_persons.py [--inbox PATH]
"""
# ruff: noqa: E501  (the seed table keeps one mandate per line)

from __future__ import annotations

import argparse
import io
import sys
import unicodedata
from datetime import date

import pandas as pd
import yaml

from kpr.collect.common import (
    CONFIG_DIR,
    cache_key,
    cached_get_json,
    http_session,
    load_params,
    load_universe,
)
from kpr.utils.io import REPO_ROOT, raw_dir

API = "https://www.wikidata.org/w/api.php"
TOLERANCE_DAYS = 31

# person_id | enwiki title (empty: none) | company_id | role | start | end | founder | co_ceo
# | note | qid (optional: Wikidata item when there is no English page)
SEED = """
elon_musk|Elon_Musk|TSLA|CEO|2008-10-01||1|0|
elon_musk|Elon_Musk|TSLA|chair|2004-04-01|2018-11-12|1|0|SEC settlement: left the chair in Nov 2018
jensen_huang|Jensen_Huang|NVDA|CEO|1993-04-05||1|0|
mark_zuckerberg|Mark_Zuckerberg|META|CEO|2004-02-04||1|0|
jeff_bezos|Jeff_Bezos|AMZN|CEO|1994-07-05|2021-07-04|1|0|
jeff_bezos|Jeff_Bezos|AMZN|chair|2021-07-05||1|0|executive chair
andy_jassy|Andy_Jassy|AMZN|CEO|2021-07-05||0|0|
tim_cook|Tim_Cook|AAPL|CEO|2011-08-24||0|0|
steve_ballmer|Steve_Ballmer|MSFT|CEO|2000-01-13|2014-02-03|0|0|
satya_nadella|Satya_Nadella|MSFT|CEO|2014-02-04||0|0|
larry_page|Larry_Page|GOOGL|CEO|2011-04-04|2019-12-02|1|0|Google CEO, then Alphabet CEO from 2015-10-02
larry_page|Larry_Page|GOOGL|other|2019-12-03||1|0|board member, controlling shareholder
sergey_brin|Sergey_Brin|GOOGL|other|1998-09-04||1|0|president of Alphabet 2015-2019, then board member
sundar_pichai|Sundar_Pichai|GOOGL|other|2015-10-02|2019-12-02|0|0|CEO of the Google subsidiary
sundar_pichai|Sundar_Pichai|GOOGL|CEO|2019-12-03||0|0|
reed_hastings|Reed_Hastings|NFLX|CEO|1998-01-01|2023-01-18|1|1|co-CEO with Sarandos from 2020-07-16
reed_hastings|Reed_Hastings|NFLX|chair|2023-01-19||1|0|executive chair
ted_sarandos|Ted_Sarandos|NFLX|CEO|2020-07-16||0|1|co-CEO
greg_peters|Greg_Peters_(businessman)|NFLX|CEO|2023-01-19||0|1|co-CEO
warren_buffett|Warren_Buffett|BRK|CEO|1970-05-10|2025-12-31|0|0|
greg_abel|Greg_Abel|BRK|other|2018-01-10|2025-12-31|0|0|vice chair, designated successor (2021)
greg_abel|Greg_Abel|BRK|CEO|2026-01-01||0|0|outside the study period
jamie_dimon|Jamie_Dimon|JPM|CEO|2005-12-31||0|0|
larry_ellison|Larry_Ellison|ORCL|CEO|1977-06-16|2014-09-17|1|0|
larry_ellison|Larry_Ellison|ORCL|chair|2014-09-18||1|0|executive chair and CTO
safra_catz|Safra_Catz|ORCL|CEO|2014-09-18|2025-09-21|0|1|co-CEO with Mark Hurd until 2019-09
mark_hurd|Mark_Hurd|ORCL|CEO|2014-09-18|2019-09-10|0|1|co-CEO, medical leave then death (2019-10-18)
alex_karp|Alex_Karp|PLTR|CEO|2004-01-01||1|0|
peter_thiel|Peter_Thiel|PLTR|chair|2003-05-01||1|0|
brian_armstrong|Brian_Armstrong_(businessman)|COIN|CEO|2012-06-01||1|0|
michael_saylor|Michael_J._Saylor|MSTR|CEO|1989-11-17|2022-08-07|1|0|
michael_saylor|Michael_J._Saylor|MSTR|chair|2022-08-08||1|0|executive chair
phong_le||MSTR|CEO|2022-08-08||0|0|no English page (redirects to the company)
lisa_su|Lisa_Su|AMD|CEO|2014-10-08||0|0|
rory_read|Rory_Read|AMD|CEO|2011-08-25|2014-10-07|0|0|
marc_benioff|Marc_Benioff|CRM|CEO|1999-03-08||1|1|co-CEO periods with Keith Block and Bret Taylor
bob_iger|Bob_Iger|DIS|CEO|2005-10-01|2020-02-24|0|0|
bob_iger|Bob_Iger|DIS|other|2020-02-25|2021-12-31|0|0|executive chair
bob_iger|Bob_Iger|DIS|CEO|2022-11-20||0|0|return as CEO
bob_chapek|Bob_Chapek|DIS|CEO|2020-02-25|2022-11-19|0|0|ousted
dennis_muilenburg|Dennis_Muilenburg|BA|CEO|2015-07-01|2019-12-22|0|0|ousted after the 737 MAX crashes
dave_calhoun|Dave_Calhoun|BA|CEO|2020-01-13|2024-08-07|0|0|
kelly_ortberg|Kelly_Ortberg|BA|CEO|2024-08-08||0|0|
howard_schultz|Howard_Schultz|SBUX|CEO|2008-01-07|2017-04-02|0|0|second stint
kevin_johnson|Kevin_Johnson_(American_businessman)|SBUX|CEO|2017-04-03|2022-04-03|0|0|
howard_schultz|Howard_Schultz|SBUX|CEO|2022-04-04|2023-03-19|0|0|interim CEO
laxman_narasimhan|Laxman_Narasimhan|SBUX|CEO|2023-03-20|2024-08-12|0|0|ousted
brian_niccol|Brian_Niccol|SBUX|CEO|2024-09-09||0|0|
ryan_cohen|Ryan_Cohen|GME|chair|2021-06-09||0|0|board member from 2021-01-11
ryan_cohen|Ryan_Cohen|GME|CEO|2023-09-28||0|0|
george_sherman||GME|CEO|2019-04-15|2021-06-20|0|0|no English page
paul_raines||GME|CEO|2010-06-01|2019-04-14|0|0|no English page
matt_furlong||GME|CEO|2021-06-21|2023-06-07|0|0|no English page
jack_dorsey|Jack_Dorsey|XYZ|CEO|2009-02-01||1|0|
tobi_lutke|Tobias_Lütke|SHOP|CEO|2008-01-01||1|0|
brian_chesky|Brian_Chesky|ABNB|CEO|2008-08-01||1|0|
evan_spiegel|Evan_Spiegel|SNAP|CEO|2012-01-01||1|0|
brian_krzanich|Brian_Krzanich|INTC|CEO|2013-05-16|2018-06-20|0|0|resigned
bob_swan|Bob_Swan|INTC|CEO|2018-06-21|2021-02-14|0|0|interim until 2019-01-31
pat_gelsinger|Pat_Gelsinger|INTC|CEO|2021-02-15|2024-11-30|0|0|forced retirement
michelle_johnston_holthaus||INTC|CEO|2024-12-01|2025-03-17|0|1|interim co-CEO, no English page|Q131981722
lip_bu_tan|Lip-Bu_Tan|INTC|CEO|2025-03-18||0|0|
larry_fink|Larry_Fink|BLK|CEO|1988-01-01||1|0|
lloyd_blankfein|Lloyd_Blankfein|GS|CEO|2006-06-01|2018-09-30|0|0|
david_solomon|David_M._Solomon|GS|CEO|2018-10-01||0|0|homonym DJ (dossier §4.5)
vlad_tenev|Vlad_Tenev|HOOD|CEO|2013-04-01||1|0|
bernard_arnault|Bernard_Arnault|MC|CEO|1989-01-13||0|0|controlling shareholder
christophe_de_margerie|Christophe_de_Margerie|TTE|CEO|2010-02-12|2014-10-20|0|0|died in a plane crash
patrick_pouyanne|Patrick_Pouyanné|TTE|CEO|2014-10-22||0|0|
francois_henri_pinault|François-Henri_Pinault|KER|CEO|2005-03-01|2025-09-14|0|0|controlling family
luca_de_meo|Luca_de_Meo|KER|CEO|2025-09-15||0|0|
carlos_ghosn|Carlos_Ghosn|RNO|CEO|2005-04-29|2019-01-23|0|0|arrested 2018-11-19
thierry_bollore|Thierry_Bolloré|RNO|CEO|2019-01-24|2019-10-10|0|0|interim from 2018-11-20, ousted
clotilde_delbos|Clotilde_Delbos|RNO|CEO|2019-10-11|2020-06-30|0|0|interim
luca_de_meo|Luca_de_Meo|RNO|CEO|2020-07-01|2025-07-15|0|0|
francois_provost|François_Provost_(CEO)|RNO|CEO|2025-07-31||0|0|
sergio_marchionne|Sergio_Marchionne|STLAM|CEO|2004-06-01|2018-07-20|0|0|FCA, replaced days before his death
mike_manley||STLAM|CEO|2018-07-21|2021-01-15|0|0|FCA|Q55698517
carlos_tavares|Carlos_Tavares|STLAM|CEO|2021-01-16|2024-12-01|0|0|resigned
antonio_filosa|Antonio_Filosa|STLAM|CEO|2025-06-23||0|0|
martin_winterkorn|Martin_Winterkorn|VOW3|CEO|2007-01-01|2015-09-23|0|0|resigned after Dieselgate
matthias_muller|Matthias_Müller_(businessman)|VOW3|CEO|2015-09-25|2018-04-12|0|0|
herbert_diess|Herbert_Diess|VOW3|CEO|2018-04-13|2022-08-31|0|0|ousted
oliver_blume|Oliver_Blume|VOW3|CEO|2022-09-01||0|0|
michael_oleary|Michael_O'Leary_(businessman)|RYA|CEO|1994-01-01||0|0|Group CEO of Ryanair Holdings
yves_guillemot|Yves_Guillemot|UBI|CEO|1988-01-01||1|0|
franck_riboud|Franck_Riboud|BN|CEO|1996-05-01|2014-09-30|0|0|
emmanuel_faber|Emmanuel_Faber|BN|CEO|2014-10-01|2021-03-15|0|0|ousted
antoine_de_saint_affrique|Antoine_de_Saint-Affrique|BN|CEO|2021-09-15||0|0|
lars_rebien_sorensen|Lars_Rebien_Sørensen|NOVO|CEO|2000-11-13|2016-12-31|0|0|
lars_fruergaard_jorgensen|Lars_Fruergaard_Jørgensen|NOVO|CEO|2017-01-01|2025-08-06|0|0|ousted
mike_doustdar|Mike_Doustdar|NOVO|CEO|2025-08-07||0|0|
bob_dudley|Bob_Dudley|BP|CEO|2010-10-01|2020-02-04|0|0|
bernard_looney|Bernard_Looney|BP|CEO|2020-02-05|2023-09-11|0|0|resigned
murray_auchincloss|Murray_Auchincloss|BP|CEO|2023-09-12|2025-12-01|0|0|interim until 2024-01-17
daniel_ek|Daniel_Ek|SPOT|CEO|2006-04-23|2025-12-31|1|0|executive chair from 2026
bill_mcdermott|Bill_McDermott|SAP|CEO|2010-02-07|2019-10-10|0|1|co-CEO with Snabe until 2014-05-21
jennifer_morgan|Jennifer_Morgan|SAP|CEO|2019-10-11|2020-04-20|0|1|co-CEO
christian_klein|Christian_Klein|SAP|CEO|2019-10-11||0|1|co-CEO until 2020-04-20
peter_wennink|Peter_Wennink|ASML|CEO|2013-07-01|2024-04-23|0|0|
christophe_fouquet||ASML|CEO|2024-04-24||0|0|no English page|Q138786664
tom_enders|Tom_Enders|AIR|CEO|2012-06-01|2019-04-09|0|0|
guillaume_faury|Guillaume_Faury|AIR|CEO|2019-04-10||0|0|
sergio_ermotti|Sergio_Ermotti|UBSG|CEO|2011-11-15|2020-10-31|0|0|
ralph_hamers|Ralph_Hamers|UBSG|CEO|2020-11-01|2023-04-04|0|0|
sergio_ermotti|Sergio_Ermotti|UBSG|CEO|2023-04-05||0|0|return after the Credit Suisse takeover
ugur_sahin|Uğur_Şahin|BNTX|CEO|2008-06-02||1|0|
anshu_jain|Anshu_Jain|DBK|CEO|2012-06-01|2015-06-30|0|1|co-CEO
jurgen_fitschen|Jürgen_Fitschen|DBK|CEO|2012-06-01|2016-05-19|0|1|co-CEO
john_cryan|John_Cryan|DBK|CEO|2015-07-01|2018-04-07|0|1|co-CEO with Fitschen until 2016-05-19
christian_sewing|Christian_Sewing|DBK|CEO|2018-04-08||0|0|
a_g_lafley|A._G._Lafley|PG|CEO|2013-05-23|2015-10-31|0|0|
david_taylor|David_S._Taylor|PG|CEO|2015-11-01|2021-10-31|0|0|
jon_moeller|Jon_Moeller|PG|CEO|2021-11-01|2025-12-31|0|0|
alex_gorsky|Alex_Gorsky|JNJ|CEO|2012-04-26|2022-01-02|0|0|
joaquin_duato|Joaquin_Duato|JNJ|CEO|2022-01-03||0|0|
muhtar_kent|Muhtar_Kent|KO|CEO|2008-07-01|2017-04-30|0|0|
james_quincey|James_Quincey|KO|CEO|2017-05-01||0|0|
charles_scharf|Charles_Scharf|V|CEO|2012-11-01|2016-11-30|0|0|
alfred_kelly|Alfred_F._Kelly_Jr.|V|CEO|2016-12-01|2023-01-31|0|0|
ryan_mcinerney|Ryan_McInerney|V|CEO|2023-02-01||0|0|
rex_tillerson|Rex_Tillerson|XOM|CEO|2006-01-01|2016-12-31|0|0|
darren_woods|Darren_Woods|XOM|CEO|2017-01-01||0|0|
frank_blake|Frank_Blake_(businessman)|HD|CEO|2007-01-02|2014-10-31|0|0|
craig_menear|Craig_Menear|HD|CEO|2014-11-01|2022-02-28|0|0|
ted_decker|Ted_Decker|HD|CEO|2022-03-01||0|0|
indra_nooyi|Indra_Nooyi|PEP|CEO|2006-10-01|2018-10-02|0|0|
ramon_laguarta|Ramon_Laguarta|PEP|CEO|2018-10-03||0|0|
paul_bulcke|Paul_Bulcke|NESN|CEO|2008-04-10|2016-12-31|0|0|
ulf_mark_schneider|Ulf_Mark_Schneider|NESN|CEO|2017-01-01|2024-08-31|0|0|ousted
laurent_freixe|Laurent_Freixe|NESN|CEO|2024-09-01|2025-09-01|0|0|dismissed
philipp_navratil|Philipp_Navratil|NESN|CEO|2025-09-02||0|0|
paul_polman|Paul_Polman|ULVR|CEO|2009-01-01|2018-12-31|0|0|
alan_jope|Alan_Jope|ULVR|CEO|2019-01-01|2023-06-30|0|0|
hein_schumacher|Hein_Schumacher|ULVR|CEO|2023-07-01|2025-02-28|0|0|replaced abruptly
fernando_fernandez||ULVR|CEO|2025-03-01||0|0|no English page found
joe_kaeser|Joe_Kaeser|SIE|CEO|2013-08-01|2021-02-02|0|0|
roland_busch|Roland_Busch|SIE|CEO|2021-02-03||0|0|
jean_paul_agon|Jean-Paul_Agon|OR|CEO|2006-04-25|2021-04-30|0|0|
nicolas_hieronimus|Nicolas_Hieronimus|OR|CEO|2021-05-01||0|0|
jean_pascal_tricoire|Jean-Pascal_Tricoire|SU|CEO|2006-05-03|2023-05-03|0|0|
peter_herweck||SU|CEO|2023-05-04|2024-11-12|0|0|dismissed, no English page|Q107420160
olivier_blum||SU|CEO|2024-11-13||0|0|no English page|Q140697703
chris_viehbacher|Chris_Viehbacher|SAN|CEO|2008-12-01|2014-10-29|0|0|dismissed
olivier_brandicourt|Olivier_Brandicourt|SAN|CEO|2015-04-02|2019-08-31|0|0|
paul_hudson|Paul_Hudson_(businessman)|SAN|CEO|2019-09-01||0|0|
michael_diekmann|Michael_Diekmann|ALV|CEO|2003-04-29|2015-05-06|0|0|
oliver_bate|Oliver_Bäte|ALV|CEO|2015-05-07||0|0|
"""

SEED_COLUMNS = ["person_id", "enwiki", "company_id", "role", "start", "end", "founder",
                "co_ceo", "note", "qid"]


def load_seed() -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(SEED.strip()), sep="|", header=None, names=SEED_COLUMNS,
                     dtype=str, keep_default_na=False)
    df["founder"] = df["founder"] == "1"
    df["co_ceo"] = df["co_ceo"] == "1"
    return df


def resolve_people(session, titles: list[str]) -> dict[str, dict]:
    """enwiki title -> {qid, canonical, sitelinks, label, aliases} through wbgetentities."""
    out: dict[str, dict] = {}
    titles = sorted(set(titles))
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        params = {"action": "wbgetentities", "format": "json", "sites": "enwiki",
                  "titles": "|".join(t.replace("_", " ") for t in chunk),
                  "props": "sitelinks|labels|aliases|descriptions", "languages": "en"}
        _s, payload = cached_get_json(session, "wikidata", "people_v2_" + cache_key(chunk), API,
                                      params, pause=1.0)
        for qid, ent in payload.get("entities", {}).items():
            if not qid.startswith("Q"):
                continue
            links = {k[:-4]: v["title"].replace(" ", "_")
                     for k, v in ent.get("sitelinks", {}).items() if k.endswith("wiki")}
            out[links.get("en", "")] = {
                "qid": qid, "sitelinks": links,
                "label": ent.get("labels", {}).get("en", {}).get("value", ""),
                "aliases": [a["value"] for a in ent.get("aliases", {}).get("en", [])],
                "description": ent.get("descriptions", {}).get("en", {}).get("value", ""),
            }
    # Titles that redirect or are not normalised: resolve through English Wikipedia, in batches.
    missing = [t for t in titles if t not in out]
    for i in range(0, len(missing), 50):
        chunk = missing[i:i + 50]
        params = {"action": "query", "format": "json", "redirects": 1,
                  "titles": "|".join(chunk), "prop": "pageprops", "ppprop": "wikibase_item",
                  "formatversion": 2}
        _s, payload = cached_get_json(session, "enwiki_titles", cache_key(chunk),
                                      "https://en.wikipedia.org/w/api.php", params, pause=1.0)
        query = payload["query"]
        back = {}
        for n in query.get("normalized", []):
            back[n["to"]] = n["from"]
        for r in query.get("redirects", []):
            back[r["to"]] = back.get(r["from"], r["from"])
        for page in query.get("pages", []):
            qid = page.get("pageprops", {}).get("wikibase_item")
            asked = back.get(page["title"], page["title"]).replace(" ", "_")
            if qid:
                ent = _entity(session, qid)
                ent["sitelinks"]["en"] = page["title"].replace(" ", "_")
                out[asked] = ent
    return out


def _entity(session, qid: str) -> dict:
    params = {"action": "wbgetentities", "format": "json", "ids": qid,
              "props": "sitelinks|labels|aliases|descriptions", "languages": "en",
              "sitefilter": "|".join(["enwiki", "frwiki", "dewiki", "itwiki", "nlwiki", "dawiki"])}
    _s, payload = cached_get_json(session, "wikidata", "people_v2_" + cache_key(qid), API, params,
                                  pause=1.0)
    ent = payload["entities"][qid]
    return {
        "qid": qid,
        "sitelinks": {k[:-4]: v["title"].replace(" ", "_")
                      for k, v in ent.get("sitelinks", {}).items() if k.endswith("wiki")},
        "label": ent.get("labels", {}).get("en", {}).get("value", ""),
        "aliases": [a["value"] for a in ent.get("aliases", {}).get("en", [])],
        "description": ent.get("descriptions", {}).get("en", {}).get("value", ""),
    }


def birth_dates(session, qids: list[str]) -> dict[str, tuple[str, str]]:
    """{qid: (YYYY-MM-DD, precision)} from Wikidata P569 (date of birth)."""
    out: dict[str, tuple[str, str]] = {}
    qids = sorted({q for q in qids if q})
    for i in range(0, len(qids), 10):  # full claims are large: small batches, slow pace
        chunk = qids[i:i + 10]
        params = {"action": "wbgetentities", "format": "json", "ids": "|".join(chunk),
                  "props": "claims"}
        _s, payload = cached_get_json(session, "wikidata", "claims_people_" + cache_key(chunk),
                                      API, params, pause=3.0, retries=6)
        for qid, ent in payload.get("entities", {}).items():
            for st in ent.get("claims", {}).get("P569", []):
                snak = st.get("mainsnak", {})
                if snak.get("snaktype") != "value" or st.get("rank") == "deprecated":
                    continue
                v = snak["datavalue"]["value"]
                date_ = v["time"].lstrip("+")[:10].replace("-00", "-01")
                prec = {9: "year", 10: "month", 11: "day"}.get(v.get("precision"), "other")
                if qid not in out or st.get("rank") == "preferred":
                    out[qid] = (date_, prec)
    return out


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def aliases_for(label: str, extra: list[str]) -> list[str]:
    """Lower-case name variants for GDELT matching (dossier §4.5).

    Kept: full names of at least two words that contain the family name of the label, without
    quotes or titles. This drops nicknames ("tim apple", "the zuck"), birth names and stage
    names ("dj d-sol") that would match unrelated articles.
    """
    family = _ascii(label).lower().split()[-1] if label.split() else ""
    out: list[str] = []
    for n in [label, *extra]:
        for v in (n.lower().strip(), _ascii(n).lower().strip()):
            v = " ".join(v.replace("dr. ", "").replace("mr ", "").split())
            if (v and len(v.split()) >= 2 and v not in out and family and family in _ascii(v)
                    and not any(ch in v for ch in '()"')):
                out.append(v)
    return out[:6]


def row_infos(session, seed: pd.DataFrame, people: dict) -> dict[int, dict | None]:
    """Wikidata info of each seed row: through the English title, else through `qid`."""
    out: dict[int, dict | None] = {}
    for r in seed.itertuples():
        if r.enwiki:
            out[r.Index] = people.get(r.enwiki)
        elif r.qid:
            out[r.Index] = _entity(session, r.qid)
        else:
            out[r.Index] = None
    return out


def compare(seed: pd.DataFrame, cand: pd.DataFrame, infos: dict) -> pd.DataFrame:
    rows = []
    for r in seed.itertuples():
        info = infos[r.Index]
        qid = info["qid"] if info else None
        match = cand[(cand.person_qid == qid) & (cand.company_id == r.company_id)
                     & (cand.role == r.role)] if qid else cand.iloc[0:0]
        status, detail = "not_in_wikidata", ""
        if len(match):
            best = None
            for m in match.itertuples():
                ds = _days(r.start, m.start)
                de = _days(r.end, m.end)
                score = (ds if ds is not None else 10_000) + (de if de is not None else 0)
                if best is None or score < best[0]:
                    best = (score, m, ds, de)
            _, m, ds, de = best
            ok_start = ds is not None and ds <= TOLERANCE_DAYS
            ok_end = (not r.end and not isinstance(m.end, str)) or (
                de is not None and de <= TOLERANCE_DAYS)
            status = "agrees" if ok_start and ok_end else "differs"
            detail = f"wikidata {m.start or '?'} ({m.start_precision or '-'}) .. " \
                     f"{m.end if isinstance(m.end, str) else 'null'}"
        rows.append({"idx": r.Index, "wd_qid": qid, "status": status, "detail": detail})
    return pd.DataFrame(rows).set_index("idx")


def _days(a: str, b) -> int | None:
    if not a or not isinstance(b, str) or not b:
        return None
    return abs((date.fromisoformat(a) - date.fromisoformat(b)).days)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbox", default=str(
        REPO_ROOT / "coordination" / "inbox" / "user"
        / f"{date.today().isoformat()}-data-persons-check.md"))
    args = parser.parse_args(argv)

    seed = load_seed()
    universe = {c["company_id"]: c for c in load_universe()}
    unknown = set(seed.company_id) - set(universe)
    if unknown:
        raise SystemExit(f"unknown company_id in seed: {sorted(unknown)}")
    cand = pd.read_csv(raw_dir("wikidata") / "candidates.csv", dtype=str)
    session = http_session()
    people = resolve_people(session, [t for t in seed["enwiki"] if t])
    infos = row_infos(session, seed, people)
    births = birth_dates(session, [i["qid"] for i in infos.values() if i])
    cmp = compare(seed, cand, infos)

    persons, problems = [], []
    for r in seed.itertuples():
        info = infos[r.Index]
        c = universe[r.company_id]
        name = r.person_id.replace("_", " ")
        if info is None:
            if r.enwiki:
                problems.append(f"- `{r.person_id}`: English page `{r.enwiki}` not found")
            pages = {"en": r.enwiki} if r.enwiki else {}
            aliases = aliases_for(r.enwiki.replace("_", " ") if r.enwiki else name, [])
        else:
            pages = {"en": info["sitelinks"]["en"]} if "en" in info["sitelinks"] else {}
            for lang in c["wiki_pages"]:
                if lang != "en" and lang in info["sitelinks"]:
                    pages[lang] = info["sitelinks"][lang]
            aliases = aliases_for(info["label"] or name, info["aliases"])
        status = cmp.loc[r.Index, "status"]
        born = births.get(info["qid"]) if info else None
        entry = {
            "person_id": r.person_id, "company_id": r.company_id, "role": r.role,
            "founder": bool(r.founder),
            "birth_date": born[0] if born and born[1] in ("day", "month", "year") else None,
            "start": r.start, "end": r.end or None,
            "wiki_pages": pages, "aliases": aliases,
            "wikidata": cmp.loc[r.Index, "wd_qid"],
            "source_check": {"agrees": "wikidata+draft", "differs": "draft, differs from wikidata",
                             "not_in_wikidata": "draft, not in wikidata"}[status],
        }
        if born and born[1] != "day":
            entry["birth_date_precision"] = born[1]
        if r.co_ceo:
            entry["co_ceo"] = True
        if r.note:
            entry["note"] = r.note
        persons.append(entry)

    _write_persons(persons)
    report = _report(seed, cmp, cand, infos, problems)
    with open(args.inbox, "w", encoding="utf-8") as fh:
        fh.write(report)
    no_birth = sorted({p["person_id"] for p in persons if not p["birth_date"]})
    print(f"birth dates: {seed.person_id.nunique() - len(no_birth)} persons; missing: {no_birth}")
    print(f"{len(persons)} mandates, {seed.person_id.nunique()} persons; "
          f"{(cmp.status == 'agrees').sum()} agree with Wikidata, "
          f"{(cmp.status == 'differs').sum()} differ, "
          f"{(cmp.status == 'not_in_wikidata').sum()} not in Wikidata. Check list: {args.inbox}")
    return 0


def _write_persons(persons: list[dict]) -> None:
    header = (
        "# Owner: data session (task D2). One entry per mandate; a person may have several.\n"
        "# Drafted by scripts/data_draft_persons.py (seed + Wikidata), then checked by hand.\n"
        "# Fields: person_id, company_id, role (CEO | CFO | chair | founder | other), founder,\n"
        "#   start, end (last day in the role, inclusive; null = still in office), wiki_pages\n"
        "#   (canonical titles; local language for European companies), aliases (GDELT\n"
        "#   matching, lower case), wikidata (QID), source_check, co_ceo (optional), note.\n"
        "# source_check: 'wikidata+draft' = draft dates within 31 days of Wikidata;\n"
        "#   'draft, ...' = needs a human check (list in coordination/inbox/user/).\n"
        "#   Set it to 'manual' once checked.\n"
    )
    body = yaml.safe_dump({"persons": persons}, allow_unicode=True, sort_keys=False, width=100)
    (CONFIG_DIR / "persons.yaml").write_text(header + body, encoding="utf-8")


def _report(seed, cmp, cand, infos, problems) -> str:
    params = load_params()
    p_start, p_end = params["period"]["start"], params["period"]["end"]
    lines = [
        "From: data", "To: user",
        "Need: manual check of the mandates in config/persons.yaml (task D2, dossier §4.4 "
        "step 5 and §5.6)",
        "Why: Wikidata is incomplete; dates decide succession resets and natural experiments",
        "Urgency: medium (collectors run on the draft; dates matter from D11 / Q4 on)", "",
        "How: for each line below, check the dates (Wikipedia page of the person, company press "
        "release or 8-K item 5.02), fix `start` / `end` in config/persons.yaml if needed and set "
        "`source_check: manual`. Lines are sorted by company.", "",
        "## Mandates whose draft dates differ from Wikidata", "",
    ]
    merged = seed.join(cmp)
    for r in merged[merged.status == "differs"].sort_values("company_id").itertuples():
        lines.append(f"- [ ] {r.company_id} `{r.person_id}` {r.role} draft {r.start} .. "
                     f"{r.end or 'null'}; {r.detail}")
    lines += ["", "## Mandates absent from Wikidata (draft only)", ""]
    for r in merged[merged.status == "not_in_wikidata"].sort_values("company_id").itertuples():
        lines.append(f"- [ ] {r.company_id} `{r.person_id}` {r.role} draft {r.start} .. "
                     f"{r.end or 'null'}" + (f" ({r.note})" if r.note else ""))
    lines += ["", "## Wikidata CEOs or chairs of the period missing from the seed", "",
              "Add them to the seed of scripts/data_draft_persons.py if they matter.", ""]
    known = {(infos[r.Index]["qid"], r.company_id) for r in seed.itertuples() if infos[r.Index]}
    end = pd.to_datetime(cand.end)
    start = pd.to_datetime(cand.start)
    in_period = (end.isna() | (end >= p_start)) & (start.isna() | (start <= p_end))
    extra = cand[in_period & cand.role.isin(["CEO", "chair"])]
    for r in extra.itertuples():
        if (r.person_qid, r.company_id) not in known:
            lines.append(f"- {r.company_id} {r.role} {r.person_label} ({r.person_qid}) "
                         f"{r.start or '?'} .. {r.end if isinstance(r.end, str) else 'null'}")
    if problems:
        lines += ["", "## Pages not found", ""] + problems
    lines += ["", "## Who each English page points to (spot homonyms)", ""]
    for r in seed.drop_duplicates("person_id").itertuples():
        info = infos[r.Index]
        desc = (info.get("description") or "(no description)") if info else "no Wikidata item"
        lines.append(f"- `{r.person_id}`: {info['qid'] if info else '-'}, {desc}")
    lines += ["", "When done, move this file to coordination/inbox/user/done/ or tell the data "
              "session which lines changed."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
