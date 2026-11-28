"""Owner: data session. Cleaning and quality checks (task D10, dossier §6.4, §6.7)."""

import numpy as np
import pandas as pd

from kpr.build.clean import bot_spikes, check_mandates, check_views, clean_prices, clean_wiki


def test_impossible_returns_are_removed_and_large_moves_listed():
    prices = pd.DataFrame({"company_id": "X", "date": pd.date_range("2020-01-01", periods=4),
                           "ret": [0.01, np.log(0.1), np.log(1.5), np.log(4.0)]})
    out, impossible, large = clean_prices(prices, -0.8, 2.0)
    assert out["ret"].isna().tolist() == [False, True, False, True]
    assert len(impossible) == 2 and len(large) == 1


def _views(spike_day=80, local_spike=False, n=120):
    rng = np.random.default_rng(0)
    days = pd.date_range("2020-01-01", periods=n)
    total = rng.normal(1000, 30, n)
    local = rng.normal(100, 5, n)
    total[spike_day] = 50_000
    if local_spike:
        local[spike_day] = 5_000
    return pd.DataFrame({"page_id": "p", "date": days, "views_en": total - local,
                         "views_local": local, "views_total": total})


def test_spike_without_news_or_local_echo_is_a_robot():
    v = _views()
    quiet_news = pd.Series(5.0, index=v["date"])
    mask = bot_spikes(v, quiet_news, 5.0, 2.0, 60)
    assert mask.sum() == 1 and mask.iloc[80]


def test_spike_confirmed_by_news_or_local_page_is_kept():
    v = _views()
    news = pd.Series(5.0, index=v["date"])
    news.iloc[80] = 200
    assert not bot_spikes(v, news, 5.0, 2.0, 60).any()
    v2 = _views(local_spike=True)
    assert not bot_spikes(v2, pd.Series(5.0, index=v2["date"]), 5.0, 2.0, 60).any()


def test_clean_wiki_needs_gdelt():
    v = _views()
    out, removed = clean_wiki(v, None, 5.0, 2.0, 60)
    assert removed.empty and out["views_total"].notna().all()
    gdelt = pd.DataFrame({"person_id": "p", "date": v["date"], "n_articles": 5})
    out, removed = clean_wiki(v, gdelt, 5.0, 2.0, 60)
    assert len(removed) == 1 and np.isnan(out.loc[80, "views_total"])


def test_mandate_overlap_rule_applies_to_single_holder_roles_only():
    persons = [
        {"person_id": "a", "company_id": "X", "role": "CEO", "start": "2015-01-01",
         "end": "2018-12-31"},
        {"person_id": "b", "company_id": "X", "role": "CEO", "start": "2018-06-01", "end": None},
        {"person_id": "c", "company_id": "X", "role": "other", "start": "2015-01-01",
         "end": None},
        {"person_id": "d", "company_id": "X", "role": "other", "start": "2016-01-01",
         "end": None},
    ]
    assert check_mandates(persons).status == "fail"
    persons[1]["co_ceo"] = True
    assert check_mandates(persons).status == "pass"


def test_views_check_flags_missing_days_during_mandates():
    days = pd.date_range("2020-01-01", "2020-01-20")
    views = np.where(np.arange(len(days)) < 5, np.nan, 100.0)
    wiki = pd.DataFrame({"page_id": "a", "date": days, "views_en": views, "views_local": 0.0,
                         "views_total": views})
    persons = [{"person_id": "a", "company_id": "X", "role": "CEO", "start": "2019-01-01",
                "end": None, "wiki_pages": {"en": "A"}}]
    checks = check_views(wiki, persons, 0.05, "2020-01-01", "2020-01-20")
    assert [c.status for c in checks] == ["pass", "warn"]
