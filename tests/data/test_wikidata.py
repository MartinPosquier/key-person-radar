"""Owner: data session. Parsing of Wikidata statements (task D2)."""

from kpr.collect.wikidata import parse_claims


def _statement(person, start=None, end=None, prec=11, rank="normal"):
    quals = {}
    if start:
        quals["P580"] = [{"snaktype": "value",
                          "datavalue": {"value": {"time": f"+{start}T00:00:00Z",
                                                  "precision": prec}}}]
    if end:
        quals["P582"] = [{"snaktype": "value",
                          "datavalue": {"value": {"time": f"+{end}T00:00:00Z",
                                                  "precision": 11}}}]
    return {"mainsnak": {"snaktype": "value", "datavalue": {"value": {"id": person}}},
            "qualifiers": quals, "rank": rank}


def test_parse_claims_reads_roles_dates_and_precision():
    entity = {"claims": {
        "P169": [_statement("Q1", "2014-02-04"), _statement("Q2", "2000-00-00", "2014-02-03", 9)],
        "P112": [_statement("Q3")],
        "P488": [_statement("Q4", rank="deprecated")],
    }}
    rows = parse_claims(entity, "MSFT")
    assert [(r["role"], r["person_qid"]) for r in rows] == [
        ("CEO", "Q1"), ("CEO", "Q2"), ("founder", "Q3")]
    assert rows[0]["start"] == "2014-02-04" and rows[0]["start_precision"] == "day"
    assert rows[1]["start"] == "2000-01-01" and rows[1]["start_precision"] == "year"
    assert rows[1]["end"] == "2014-02-03"
    assert rows[2]["start"] is None
