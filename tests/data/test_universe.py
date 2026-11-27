"""Owner: data session. Structure of config/universe.yaml (task D1, dossier §4.2)."""

from collections import Counter

from kpr.collect.common import load_universe


def test_universe_has_60_companies_46_media_14_control():
    companies = load_universe()
    assert len(companies) == 60
    groups = Counter(c["group"] for c in companies)
    assert groups == {"media": 46, "control": 14}


def test_market_index_matches_region():
    for c in load_universe():
        expected = "^GSPC" if c["region"] == "US" else "^STOXX"
        assert c["market_index"] == expected, c["company_id"]


def test_ids_and_tickers_are_unique():
    companies = load_universe()
    assert len({c["company_id"] for c in companies}) == 60
    assert len({c["ticker"] for c in companies}) == 60


def test_wikipedia_titles_use_underscores():
    for c in load_universe():
        for title in c["wiki_pages"].values():
            assert " " not in title, (c["company_id"], title)
