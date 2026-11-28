"""Build the headlines table for shock days and their neighbours.

Owner: data session. Task D12 in PARALLEL_SESSIONS.md §7. Dossier §8.3.

For every event of the `events` table (task Q5; the fixtures until it is merged), the target
days are the UTC shock day and its neighbours (date_utc - 1 .. date_utc + 1), so that FinBERT
runs on a few thousand days instead of ten years of news (dossier §8.3 step 6).

Sources:
- until 2023: FNSPID titles (raw/fnspid/subset.parquet, from kpr.collect.fnspid);
- 2024-2025: GDELT gives only URLs; an approximate title is rebuilt from the words of the URL
  slug (many sites put the title there). One small BigQuery query per year lists the URLs of
  the target days that mention the person (partition filter on the target days only, dry run
  and byte cap as in kpr.collect.gdelt). Skipped when GCP_PROJECT is not configured.

Only titles that contain the person's name or one of their aliases are kept (dossier §8.3),
which also removes most articles about another topic. headline_id is a hash of the URL (or
of the title when there is no URL), so reruns give stable ids.

Usage: python -m kpr.collect.headlines [--no-gdelt]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from urllib.parse import unquote, urlparse

import pandas as pd

from kpr.collect.common import append_collection_log, load_persons
from kpr.utils.io import raw_dir, read_table, write_table
from kpr.utils.logging import get_logger

log = get_logger(__name__)

NEIGHBOURS = 1
FNSPID_LAST_YEAR = 2023
MIN_SLUG_WORDS = 4
_ID_TOKEN = re.compile(r"^(?:\d+|[0-9a-f]{8,}|[a-z]?\d{5,}[a-z]?|amp|html?|php|aspx?)$")


def target_days(events: pd.DataFrame, k: int = NEIGHBOURS) -> pd.DataFrame:
    """(person_id, date) pairs: each event's UTC day and its k neighbours on each side."""
    rows = [{"person_id": e.person_id, "date": e.date_utc + pd.Timedelta(days=d)}
            for e in events.itertuples() for d in range(-k, k + 1)]
    return pd.DataFrame(rows, columns=["person_id", "date"]).drop_duplicates()


def title_from_url(url: str) -> str | None:
    """Approximate title from the URL slug, or None if the slug does not look like one."""
    path = unquote(urlparse(url).path or "")
    parts = [p for p in path.split("/") if p]
    best: list[str] = []
    for part in parts:
        part = re.sub(r"\.(s?html?|php|aspx?)$", "", part.lower())
        words = [w for w in re.split(r"[-_+.]+", part) if w and not _ID_TOKEN.match(w)]
        if len(words) > len(best):
            best = words
    if len(best) < MIN_SLUG_WORDS:
        return None
    return " ".join(best).capitalize()


def mentions(title: pd.Series, aliases: list[str]) -> pd.Series:
    """True where the title contains one of the aliases (case-insensitive, word bounds)."""
    if not aliases:
        return pd.Series(False, index=title.index)
    pattern = r"\b(?:" + "|".join(re.escape(a.lower()) for a in
                                  sorted(set(aliases), key=len, reverse=True)) + r")\b"
    return title.fillna("").str.lower().str.contains(pattern, regex=True)


def headline_id(url: str | None, title: str) -> str:
    return hashlib.sha1((url or title).encode("utf-8")).hexdigest()[:16]


def from_fnspid(subset: pd.DataFrame, days: pd.DataFrame,
                aliases: dict[str, list[str]]) -> pd.DataFrame:
    """FNSPID titles of the target days that name the person."""
    s = subset.assign(date=subset["date_utc"].dt.normalize())
    s = s[s["date"].dt.year <= FNSPID_LAST_YEAR]
    out = []
    for pid, d in days.groupby("person_id"):
        cand = s[s["date"].isin(d["date"])]
        cand = cand[mentions(cand["title"], aliases.get(pid, []))]
        if len(cand):
            out.append(pd.DataFrame({
                "person_id": pid, "date": cand["date"].values,
                "headline_id": [headline_id(u, t) for u, t in
                                zip(cand["url"], cand["title"], strict=True)],
                "title": cand["title"].values,
                "source": ("fnspid:" + cand["publisher"].fillna("unknown")).values}))
    return pd.concat(out, ignore_index=True) if out else _empty()


def from_gdelt_urls(urls: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    """Titles rebuilt from GDELT URLs (columns person_id, date, url) that name the person."""
    if urls.empty:
        return _empty()
    u = urls.assign(title=urls["url"].map(title_from_url)).dropna(subset=["title"])
    keep = pd.Series(False, index=u.index)
    for pid, g in u.groupby("person_id"):
        keep[g.index] = mentions(g["title"], aliases.get(pid, []))
    u = u[keep]
    return pd.DataFrame({"person_id": u["person_id"].values, "date": u["date"].values,
                         "headline_id": [headline_id(x, "") for x in u["url"]],
                         "title": u["title"].values, "source": "gdelt_url"})


def gdelt_urls(days: pd.DataFrame, persons: list[dict]) -> pd.DataFrame:
    """URLs of 2024-2025 target days naming each person (BigQuery, cached per year)."""
    from kpr.collect import gdelt

    out = []
    pairs = gdelt.alias_rows(persons)
    cap = int(30e9)
    for year, d in days[days["date"].dt.year > FNSPID_LAST_YEAR].groupby(days["date"].dt.year):
        path = raw_dir("gdelt") / f"headline_urls_{year}.parquet"
        if path.exists():
            out.append(pd.read_parquet(path))
            continue
        day_list = sorted(d["date"].dt.date.unique())
        sql = url_query(pairs, day_list, gdelt.TABLE)
        estimate = gdelt.dry_run_bytes(sql)
        if estimate > cap:
            raise RuntimeError(f"headline URLs {year}: {estimate / 1e9:.1f} GB > cap")
        client, bq = gdelt._client()
        df = client.query(sql, job_config=bq.QueryJobConfig(maximum_bytes_billed=cap)).result() \
            .to_dataframe()
        df["date"] = pd.to_datetime(df["date"])
        wanted = set(zip(d["person_id"], d["date"], strict=True))
        df = df[[(p, t) in wanted for p, t in zip(df["person_id"], df["date"], strict=True)]]
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        append_collection_log("gdelt", headline_urls_year=year, rows=len(df), bytes=estimate)
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["person_id", "date", "url"])


def url_query(pairs: list[tuple[str, str]], days: list, table: str) -> str:
    def lit(text: str) -> str:
        return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"

    structs = ", ".join(f"STRUCT({lit(p)} AS person_id, {lit(a)} AS alias)" for p, a in pairs)
    parts = " OR ".join(
        f"(g._PARTITIONTIME >= TIMESTAMP('{d}') AND g._PARTITIONTIME < "
        f"TIMESTAMP_ADD(TIMESTAMP('{d}'), INTERVAL 1 DAY))" for d in days)
    return f"""
SELECT DISTINCT a.person_id,
  PARSE_DATE('%Y%m%d', SUBSTR(CAST(g.DATE AS STRING), 1, 8)) AS date,
  g.DocumentIdentifier AS url
FROM `{table}` AS g, UNNEST(SPLIT(LOWER(g.Persons), ';')) AS name
JOIN UNNEST([{structs}]) AS a ON TRIM(name) = a.alias
WHERE {parts}
"""


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["person_id", "date", "headline_id", "title", "source"])


def build(events: pd.DataFrame, persons: list[dict], fnspid: pd.DataFrame | None,
          urls: pd.DataFrame | None) -> pd.DataFrame:
    days = target_days(events)
    aliases: dict[str, list[str]] = {}
    for p in persons:
        aliases.setdefault(p["person_id"], [])
        aliases[p["person_id"]] += [a for a in (p.get("aliases") or [])
                                    if a not in aliases[p["person_id"]]]
    parts = []
    if fnspid is not None:
        parts.append(from_fnspid(fnspid, days, aliases))
    if urls is not None:
        parts.append(from_gdelt_urls(urls, aliases))
    df = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(
        len(p) for p in parts) else _empty()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["person_id", "date", "headline_id"])
    return df.sort_values(["person_id", "date", "headline_id"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kpr.collect.headlines")
    parser.add_argument("--no-gdelt", action="store_true", help="FNSPID titles only")
    args = parser.parse_args(argv)

    events = read_table("events")
    persons = load_persons()
    sub_path = raw_dir("fnspid") / "subset.parquet"
    fnspid = pd.read_parquet(sub_path) if sub_path.exists() else None
    if fnspid is None:
        log.warning("no FNSPID subset (run kpr.collect.fnspid --download --subset): "
                    "titles until 2023 are missing")
    urls = None
    if not args.no_gdelt:
        try:
            urls = gdelt_urls(target_days(events), persons)
        except RuntimeError as exc:
            log.warning("no GDELT URLs for 2024-2025: %s", exc)
    df = build(events, persons, fnspid, urls)
    write_table(df, "headlines")
    days = target_days(events)
    covered = df.groupby(["person_id", "date"]).ngroups if len(df) else 0
    append_collection_log("headlines", rows=len(df), target_days=len(days),
                          days_with_titles=covered)
    log.info("headlines: %d titles on %d of %d target days (%s)", len(df), covered, len(days),
             df["source"].str.split(":").str[0].value_counts().to_dict() if len(df) else {})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
