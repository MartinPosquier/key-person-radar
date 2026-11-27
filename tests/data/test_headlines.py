"""Owner: data session. Headlines of shock days (task D12, dossier §8.3)."""

import pandas as pd

from kpr.collect.headlines import build, target_days, title_from_url, url_query
from kpr.schemas import validate

PERSONS = [{"person_id": "jensen_huang", "aliases": ["jensen huang", "jen-hsun huang"]}]


def test_target_days_are_the_shock_day_and_its_neighbours():
    events = pd.DataFrame({"person_id": ["a"], "date_utc": pd.to_datetime(["2020-03-10"])})
    assert target_days(events)["date"].dt.day.tolist() == [9, 10, 11]


def test_title_from_url_slug():
    url = "https://www.reuters.com/technology/nvidia-ceo-jensen-huang-says-demand-is-insane-2024-03-19/"
    assert title_from_url(url).startswith("Nvidia ceo jensen huang says demand")
    assert title_from_url("https://example.com/article/123456789") is None


def test_build_keeps_titles_naming_the_person_on_target_days_only():
    events = pd.DataFrame({"person_id": ["jensen_huang"],
                           "date_utc": pd.to_datetime(["2023-05-25"])})
    fnspid = pd.DataFrame({
        "title": ["Jensen Huang unveils new chip", "Nvidia earnings beat",
                  "Jensen Huang interview"],
        "ticker": ["NVDA", "NVDA", "NVDA"],
        "url": ["u1", "u2", "u3"], "publisher": ["Benzinga", "Benzinga", None],
        "date_utc": pd.to_datetime(["2023-05-24 10:00", "2023-05-25 12:00", "2023-06-30 09:00"]),
    })
    urls = pd.DataFrame({"person_id": ["jensen_huang"], "date": pd.to_datetime(["2024-02-22"]),
                         "url": ["https://x.com/business/jensen-huang-nvidia-results-ai-boom"]})
    df = build(events, PERSONS, fnspid, urls)
    validate(df, "headlines")
    assert df["title"].tolist() == ["Jensen Huang unveils new chip",
                                    "Jensen huang nvidia results ai boom"]
    assert df["source"].tolist() == ["fnspid:Benzinga", "gdelt_url"]


def test_url_query_filters_on_target_day_partitions():
    sql = url_query([("a", "x y")], ["2024-01-02"], "t")
    assert "_PARTITIONTIME >= TIMESTAMP('2024-01-02')" in sql and "SELECT *" not in sql
