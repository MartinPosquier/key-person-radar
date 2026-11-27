"""Shared helpers for the data session: configuration loading, HTTP session, raw cache.

Owner: data session. Every collector reads the universe and persons from here, so that the
YAML layout is parsed in one place only.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
import yaml

from kpr.utils.io import REPO_ROOT, raw_dir

CONFIG_DIR = REPO_ROOT / "config"

REGIONS = ("US", "EU")
GROUPS = ("media", "control")
EXCHANGES = ("XNYS", "XNAS", "XPAR", "XETR", "XAMS", "XLON", "XMIL", "XSWX", "XCSE", "XDUB")
ROLES = ("CEO", "CFO", "chair", "founder", "other")
COMPANY_FIELDS = (
    "company_id", "name", "ticker", "region", "exchange", "market_index", "sector",
    "wiki_pages", "trends_entity", "group",
)


def load_yaml(name: str) -> Any:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_params() -> dict:
    return load_yaml("params/data.yaml")


def load_universe() -> list[dict]:
    """Return the list of companies of config/universe.yaml, validated."""
    companies = load_yaml("universe.yaml")["companies"]
    check_universe(companies)
    return companies


def load_persons() -> list[dict]:
    """Return the list of mandates of config/persons.yaml (one entry per mandate)."""
    return load_yaml("persons.yaml").get("persons") or []


def check_universe(companies: list[dict]) -> None:
    """Raise ValueError if an entry of universe.yaml breaks the documented layout."""
    problems: list[str] = []
    seen: set[str] = set()
    for i, c in enumerate(companies):
        cid = c.get("company_id", f"#{i}")
        missing = [f for f in COMPANY_FIELDS if f not in c]
        if missing:
            problems.append(f"{cid}: missing {missing}")
        if cid in seen:
            problems.append(f"{cid}: duplicate company_id")
        seen.add(cid)
        if c.get("region") not in REGIONS:
            problems.append(f"{cid}: region {c.get('region')!r}")
        if c.get("group") not in GROUPS:
            problems.append(f"{cid}: group {c.get('group')!r}")
        if c.get("exchange") not in EXCHANGES:
            problems.append(f"{cid}: exchange {c.get('exchange')!r}")
        pages = c.get("wiki_pages") or {}
        if "en" not in pages:
            problems.append(f"{cid}: no English Wikipedia page")
    if problems:
        raise ValueError("universe.yaml: " + "; ".join(problems))


def user_agent() -> str:
    """Descriptive User-Agent required by Wikimedia and SEC EDGAR (dossier §5.2, §5.6)."""
    contact = os.environ.get("HTTP_CONTACT", "").strip()
    base = "KeyPersonRiskRadar/0.1 (student research project, non-commercial"
    return f"{base}; {contact})" if contact else f"{base})"


def http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent(), "Accept-Encoding": "gzip"})
    return session


def cache_key(*parts: Any) -> str:
    """Stable file-name-safe key for a raw response."""
    text = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def cached_get_json(
    session: requests.Session,
    source: str,
    key: str,
    url: str,
    params: dict | None = None,
    pause: float = 0.0,
    retries: int = 4,
    ok_statuses: tuple[int, ...] = (200,),
) -> tuple[int, Any]:
    """GET `url` once and keep the raw JSON in raw/<source>/<key>.json (dossier §5.9).

    Returns (status, payload). A cached response is never downloaded again. Statuses in
    `ok_statuses` are cached (e.g. 404 for a page without views); 429 and 5xx are retried
    with exponential back-off.
    """
    path = raw_dir(source) / f"{key}.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        return record["status"], record["payload"]
    delay = 2.0
    for attempt in range(retries + 1):
        resp = session.get(url, params=params, timeout=60)
        if resp.status_code in ok_statuses:
            try:
                payload = resp.json()
            except ValueError:
                payload = resp.text
            if isinstance(payload, dict) and "error" in payload and resp.status_code == 200:
                # MediaWiki APIs report bad requests with HTTP 200: never cache them.
                raise RuntimeError(f"API error for {resp.url}: {payload['error']}")
            _write_json_atomic(path, {"status": resp.status_code, "url": resp.url,
                                      "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                      "payload": payload})
            if pause:
                time.sleep(pause)
            return resp.status_code, payload
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            wait = delay
            retry_after = resp.headers.get("Retry-After", "")
            if retry_after.isdigit():
                wait = max(wait, min(float(retry_after), 300.0))
            time.sleep(wait)
            delay *= 2
            continue
        resp.raise_for_status()
    raise RuntimeError(f"unreachable: {url}")


def _write_json_atomic(path: Path, record: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def append_collection_log(source: str, **fields: Any) -> None:
    """One line per collection run: date, parameters, rows, errors (dossier §5.9)."""
    line = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}
    with open(raw_dir(source) / "_collection_log.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, default=str) + "\n")
