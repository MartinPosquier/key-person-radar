"""Aggregate GDELT GKG per person and day via BigQuery (dry run first).

Owner: data session. Task D5, D8 in PARALLEL_SESSIONS.md §7. Dossier §5.3, §6.4.

Economy rules of dossier §5.3, all enforced here:

1. the partitioned table is filtered on its partition column (`_PARTITIONTIME`);
2. only four columns are read: DATE, Persons, V2Tone, DocumentIdentifier;
3. `--dry-run` estimates the bytes of one month and extrapolates to the whole period before
   anything is run for real;
4. one query per year for all persons at once, aggregated per person and day in BigQuery, so
   that only a small table is downloaded;
5. every real query carries `maximum_bytes_billed`, so BigQuery refuses it rather than bill
   more than the dry run announced (set a billing cap at 0 in the project as well).

Commands:
  python -m kpr.collect.gdelt --dry-run 2019-06    # D5: estimate, extrapolate, apply the rule
  python -m kpr.collect.gdelt --year 2019          # D8: one real query, cached in raw/gdelt/
  python -m kpr.collect.gdelt --build              # D8: gdelt_daily table from the cache

Needs GCP_PROJECT in .env and Google application-default credentials
(`gcloud auth application-default login`).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd

from kpr.collect.common import append_collection_log, load_params, load_persons
from kpr.utils.io import raw_dir, write_table
from kpr.utils.logging import get_logger

log = get_logger(__name__)

TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"
GKG_START = "2015-02-19"  # first day of GDELT 2.0 GKG
TIB = 1024**4

QUERY = """
WITH aliases AS (
  SELECT person_id, alias
  FROM UNNEST([{aliases}]) AS a
),
mentions AS (
  SELECT DISTINCT
    a.person_id,
    PARSE_DATE('%Y%m%d', SUBSTR(CAST(g.DATE AS STRING), 1, 8)) AS day,
    g.DocumentIdentifier AS url,
    SAFE_CAST(SPLIT(g.V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64) AS tone
  FROM `{table}` AS g, UNNEST(SPLIT(LOWER(g.Persons), ';')) AS name
  JOIN aliases AS a ON TRIM(name) = a.alias
  WHERE g._PARTITIONTIME >= TIMESTAMP('{start}')
    AND g._PARTITIONTIME < TIMESTAMP('{end}')
)
SELECT
  person_id,
  day AS date,
  COUNT(*) AS n_articles,
  AVG(tone) AS tone_mean,
  COUNTIF(tone < 0) / COUNT(tone) AS tone_neg_share,
  STDDEV_SAMP(tone) AS tone_std
FROM mentions
GROUP BY person_id, day
"""


def alias_rows(persons: list[dict]) -> list[tuple[str, str]]:
    """Unique (person_id, alias) pairs; an alias shared by two persons is dropped (§4.5)."""
    pairs = {(p["person_id"], a.strip().lower())
             for p in persons for a in (p.get("aliases") or []) if a.strip()}
    owners: dict[str, set[str]] = {}
    for pid, alias in pairs:
        owners.setdefault(alias, set()).add(pid)
    ambiguous = {a for a, o in owners.items() if len(o) > 1}
    if ambiguous:
        log.warning("aliases shared by several persons, dropped: %s", sorted(ambiguous))
    return sorted((pid, a) for pid, a in pairs if a not in ambiguous)


def build_query(pairs: list[tuple[str, str]], start: str, end: str) -> str:
    def lit(text: str) -> str:
        return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"

    structs = ", ".join(f"STRUCT({lit(pid)} AS person_id, {lit(a)} AS alias)" for pid, a in pairs)
    return QUERY.format(aliases=structs, table=TABLE, start=start, end=end)


def extrapolate(month_bytes: int, month: str, period_start: str, period_end: str) -> float:
    """Scale one month's scanned bytes to the whole period, pro rata of days."""
    m0 = pd.Timestamp(month + "-01")
    days_month = (m0 + pd.offsets.MonthBegin(1) - m0).days
    days_total = (pd.Timestamp(period_end) - pd.Timestamp(period_start)).days + 1
    return month_bytes * days_total / days_month


def decide(total_bytes: float, free_quota_bytes: float, max_share: float) -> str:
    """Rule of task D5: under max_share of the free quota -> GDELT, otherwise FNSPID."""
    return "GDELT" if total_bytes < max_share * free_quota_bytes else "FNSPID"


def _client():
    project = os.environ.get("GCP_PROJECT", "").strip()
    if not project:
        raise RuntimeError("GCP_PROJECT is empty in .env (see coordination/inbox/user/)")
    from google.cloud import bigquery

    return bigquery.Client(project=project), bigquery


def dry_run_bytes(sql: str) -> int:
    client, bq = _client()
    job = client.query(sql, job_config=bq.QueryJobConfig(dry_run=True, use_query_cache=False))
    return int(job.total_bytes_processed)


def run_year(year: int, pairs: list[tuple[str, str]], max_bytes: int) -> pd.DataFrame:
    path = raw_dir("gdelt") / f"{year}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    start = max(f"{year}-01-01", GKG_START)
    sql = build_query(pairs, start, f"{year + 1}-01-01")
    estimate = dry_run_bytes(sql)
    if estimate > max_bytes:
        raise RuntimeError(f"{year}: dry run says {estimate / 1e9:.1f} GB > cap "
                           f"{max_bytes / 1e9:.1f} GB, not run")
    client, bq = _client()
    job = client.query(sql, job_config=bq.QueryJobConfig(maximum_bytes_billed=max_bytes))
    df = job.result().to_dataframe()
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    append_collection_log("gdelt", year=year, rows=len(df), bytes_estimated=estimate,
                          bytes_billed=job.total_bytes_billed, aliases=len(pairs))
    return df


def build_daily(raw: pd.DataFrame, person_ids: list[str], start: str, end: str,
                min_articles_for_tone: int) -> pd.DataFrame:
    """Complete person x day grid: days without articles get n_articles = 0 and no tone."""
    days = pd.date_range(start, end, freq="D")
    grid = pd.MultiIndex.from_product([sorted(person_ids), days], names=["person_id", "date"])
    raw = raw.assign(date=pd.to_datetime(raw["date"]))
    df = raw.set_index(["person_id", "date"]).reindex(grid).reset_index()
    df["n_articles"] = df["n_articles"].fillna(0).astype("int64")
    thin = df["n_articles"] < min_articles_for_tone  # dossier §6.4: no tone under 3 articles
    for col in ("tone_mean", "tone_neg_share", "tone_std"):
        df[col] = df[col].astype(float)
        df.loc[thin, col] = float("nan")
    return df[["person_id", "date", "n_articles", "tone_mean", "tone_neg_share", "tone_std"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kpr.collect.gdelt")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", metavar="YYYY-MM", help="estimate one month and extrapolate")
    g.add_argument("--year", type=int, nargs="+", help="run the real query for these years")
    g.add_argument("--build", action="store_true", help="write gdelt_daily from raw/gdelt/")
    g.add_argument("--print-sql", metavar="YYYY-MM", help="print the query of one month")
    args = parser.parse_args(argv)

    params = load_params()
    gp = params["gdelt"]
    period_start = max(params["period"]["start"], GKG_START)
    period_end = params["period"]["end"]
    persons = load_persons()
    pairs = alias_rows(persons)

    if args.print_sql or args.dry_run:
        month = args.print_sql or args.dry_run
        m0 = pd.Timestamp(month + "-01")
        sql = build_query(pairs, m0.date().isoformat(),
                          (m0 + pd.offsets.MonthBegin(1)).date().isoformat())
        if args.print_sql:
            print(sql)
            return 0
        month_bytes = dry_run_bytes(sql)
        total = extrapolate(month_bytes, month, period_start, period_end)
        quota = gp.get("free_quota_tib", 1.0) * TIB
        choice = decide(total, quota, gp["max_share_of_free_quota"])
        result = {"month": month, "month_bytes": month_bytes, "total_bytes": total,
                  "free_quota_bytes": quota, "share_of_quota": total / quota,
                  "aliases": len(pairs), "decision": choice,
                  "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (raw_dir("gdelt") / f"dry_run_{month}.json").write_text(json.dumps(result, indent=2))
        append_collection_log("gdelt", **result)
        print(json.dumps(result, indent=2))
        return 0

    if args.year:
        cap = int(gp.get("max_bytes_per_query_gb", 200) * 1e9)
        for year in args.year:
            df = run_year(year, pairs, cap)
            log.info("%d: %d person-days", year, len(df))
        return 0

    files = sorted(raw_dir("gdelt").glob("[0-9][0-9][0-9][0-9].parquet"))
    if not files:
        raise RuntimeError("no raw/gdelt/<year>.parquet yet: run --year first")
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    daily = build_daily(raw, sorted({p["person_id"] for p in persons}), period_start,
                        period_end, gp["min_articles_for_tone"])
    write_table(daily, "gdelt_daily")
    log.info("gdelt_daily: %d rows from %d yearly files", len(daily), len(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
