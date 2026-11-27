"""Owner: data session. Wikipedia pageviews table (task D4, dossier §5.2, §6.4)."""

import numpy as np
import pandas as pd

from kpr.collect.wiki import build_table, page_list
from kpr.schemas import validate

START, END = "2020-01-01", "2020-01-05"


def _s(values, start=START):
    return pd.Series(values, index=pd.date_range(start, periods=len(values)), dtype=float)


def test_english_plus_local_and_missing_days_stay_missing():
    pages = pd.DataFrame({"page_id": ["p", "p"], "lang": ["en", "fr"], "title": ["P", "P"]})
    views = {("p", "en"): _s([10, 20, np.nan, 40]), ("p", "fr"): _s([1, 2, np.nan, np.nan])}
    df = build_table(pages, views, START, END)
    validate(df, "wiki_views")
    assert df["views_total"].tolist()[:2] == [11.0, 22.0]
    assert np.isnan(df.loc[2, "views_total"])  # no data in any language: missing, not zero
    assert df.loc[3, "views_total"] == 40.0  # local missing, English present
    assert np.isnan(df.loc[4, "views_en"])  # day after the series ends


def test_page_without_local_version_has_zero_local_views():
    pages = pd.DataFrame({"page_id": ["q"], "lang": ["en"], "title": ["Q"]})
    df = build_table(pages, {("q", "en"): _s([5, 5, 5, 5, 5])}, START, END)
    assert (df["views_local"] == 0).all() and (df["views_total"] == 5).all()


def test_page_list_covers_persons_and_companies():
    pages = page_list()
    assert pages["page_id"].str.startswith("company_").sum() >= 60
    assert "jensen_huang" in set(pages["page_id"])
    assert not pages.duplicated(["page_id", "lang"]).any()
