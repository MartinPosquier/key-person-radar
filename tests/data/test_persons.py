"""Owner: data session. config/persons.yaml and the mandates table (task D2, dossier §4.4, §6.7)."""

import pandas as pd

from kpr.collect.common import ROLES, load_persons, load_universe
from kpr.collect.wikidata import mandates_from_persons
from kpr.schemas import validate


def test_every_company_has_at_least_one_mandate_and_ids_are_known():
    persons = load_persons()
    companies = {c["company_id"] for c in load_universe()}
    used = {p["company_id"] for p in persons}
    assert used <= companies
    assert used == companies


def test_roles_and_dates_are_valid():
    for p in load_persons():
        assert p["role"] in ROLES, p
        if p.get("end"):
            assert str(p["start"]) <= str(p["end"]), p


def test_mandates_table_matches_frozen_schema():
    validate(mandates_from_persons(load_persons()), "mandates")


def test_no_overlapping_ceo_mandates_except_declared_co_ceos():
    """Dossier §6.7: same role, same company, no overlap unless co-CEO is declared."""
    rows = [p for p in load_persons() if p["role"] == "CEO" and not p.get("co_ceo")]
    df = pd.DataFrame(rows)
    df["start"] = pd.to_datetime(df["start"].astype(str))
    df["end"] = pd.to_datetime(df["end"].astype(str).replace("None", None))
    clashes = []
    for cid, g in df.sort_values("start").groupby("company_id"):
        prev_end = None
        for r in g.itertuples():
            if prev_end is not None and (pd.isna(prev_end) or r.start <= prev_end):
                clashes.append((cid, r.person_id, r.start.date()))
            prev_end = r.end if prev_end is None or pd.isna(r.end) else max(prev_end, r.end)
    assert not clashes, clashes


def test_mandates_from_persons_parses_open_mandates():
    df = mandates_from_persons([
        {"person_id": "a", "company_id": "X", "role": "CEO", "start": "2015-01-01",
         "end": None, "founder": True},
    ])
    assert pd.isna(df.loc[0, "end"]) and bool(df.loc[0, "founder"])
