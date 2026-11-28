"""Owner: ml session. Tests for kpr.build.features (task M2, dossier §9.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kpr.build import features as F
from kpr.schemas import validate

FIXTURE_TABLES = ("events", "panel", "prices", "market", "vix", "mandates", "trends_ratio",
                  "wiki_views", "gdelt_daily", "headlines_tone")
COMPANIES = pd.DataFrame({
    "company_id": ["FAKE1", "FAKE2", "FAKE3"],
    "region": ["US", "US", "EU"],
    "sector": ["Semiconductors", "Automobiles", "Energy"],
    "market_index": ["^GSPC", "^GSPC", "^STOXX"],
})


@pytest.fixture(scope="module")
def tables(monkeypatch_module):
    monkeypatch_module.setenv("KPR_USE_FIXTURES", "1")
    from kpr.utils.io import read_table

    return {name: read_table(name) for name in FIXTURE_TABLES}


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def feats(tables):
    return F.build_features(**tables, companies=COMPANIES)


def _row(df, event_id):
    return df.set_index("event_id").loc[event_id]


def test_one_numeric_row_per_event_and_valid_schema(feats, tables):
    assert len(feats) == len(tables["events"])
    assert feats["event_id"].is_unique
    validate(feats, "features")
    assert all(pd.api.types.is_float_dtype(feats[c]) for c in feats.columns if c != "event_id")


def test_every_family_is_present(feats):
    for family in F.FAMILIES:
        assert F.family_columns(feats.columns, family), family


def test_tone_split_into_negative_and_positive_parts(feats):
    assert (feats["tone_neg"] >= 0).all() and (feats["tone_pos"] >= 0).all()
    assert np.allclose(feats["tone_pos"] - feats["tone_neg"], feats["tone_rel"])
    assert ((feats["tone_neg"] == 0) | (feats["tone_pos"] == 0)).all()


def test_one_hot_categories(feats, tables):
    roles = feats[F.family_columns(feats.columns, "C_person")].filter(like="role_")
    assert (roles.sum(axis=1) == 1).all()
    ev = tables["events"].set_index("event_id")
    eu = ev.index[ev["company_id"] == "FAKE3"]
    assert (feats.set_index("event_id").loc[eu, "region_EU"] == 1).all()


def test_person_features(feats, tables):
    ev = tables["events"].iloc[0]  # p_alpha, CEO and founder since 2005
    row = _row(feats, ev.event_id)
    assert row["founder"] == 1 and row["role_CEO"] == 1
    assert row["tenure_years"] == pytest.approx((ev.j0 - pd.Timestamp("2005-01-01")).days
                                                / 365.25)
    assert row["tone_x_founder"] == pytest.approx(ev.tone_rel)


def test_age_and_tenure_match_the_reference_fixture(feats, monkeypatch_module):
    """The frozen features fixture holds reference ceo_age and tenure_years values."""
    from kpr.utils.io import read_table

    ref = read_table("features").set_index("event_id")
    mine = feats.set_index("event_id").loc[ref.index]
    np.testing.assert_allclose(mine["tenure_years"], ref["tenure_years"])
    np.testing.assert_allclose(mine["ceo_age"], ref["ceo_age"])  # NaN where birth date unknown
    assert mine["ceo_age"].isna().any() and mine["ceo_age"].notna().any()


def test_attention_share_uses_previous_year_only(feats, tables):
    ev = tables["events"].set_index("event_id")
    trends = tables["trends_ratio"].set_index(["person_id", "year"])["ratio_person_company"]
    for event_id, row in feats.set_index("event_id").iterrows():
        e = ev.loc[event_id]
        expected = trends.get((e.person_id, e.j0.year - 1), np.nan)
        assert row["attention_share"] == pytest.approx(expected, nan_ok=True)
    # fixture trends start in 2020, so every 2020 event has no previous-year value
    assert feats.loc[ev.loc[feats["event_id"], "j0"].dt.year.values == 2020,
                     "attention_share"].isna().all()


def test_market_context(feats, tables):
    e = tables["events"].iloc[5]
    row = _row(feats, e.event_id)
    px = tables["prices"].query("company_id == @e.company_id").set_index("date")
    assert row["vol_20d"] == pytest.approx(px.loc[px.index < e.j0, "ret"].iloc[-20:].std())
    assert row["log_mcap"] == pytest.approx(
        np.log((px["close_adj"] * px["shares_out"]).loc[px.index < e.j0].iloc[-1]))
    assert row["peak_days"] >= 1  # the planted shock day is a peak


def test_peak_days_counts_only_past_days():
    z = pd.Series([0, 3, 3, 3, 0, 3], index=pd.date_range("2021-01-01", periods=6))
    assert F._peak_days(z, z.index[2], 2.0) == 2
    assert F._peak_days(z, z.index[4], 2.0) == 0


def test_no_look_ahead(tables):
    """Scrambling every input after J0 must not change the features of that event."""
    e = tables["events"].sort_values("j0").iloc[len(tables["events"]) // 2]
    base = _row(F.build_features(**tables, companies=COMPANIES), e.event_id)

    rng = np.random.default_rng(0)
    scrambled = {}
    for name, df in tables.items():
        df = df.copy()
        if name == "trends_ratio":
            df.loc[df["year"] >= e.j0.year, "ratio_person_company"] = 99.0
        elif "date" in df.columns:
            future = df["date"] > e.j0
            for col in df.select_dtypes("number").columns:
                df.loc[future, col] = rng.permutation(df.loc[future, col].to_numpy())
        scrambled[name] = df
    after = _row(F.build_features(**scrambled, companies=COMPANIES), e.event_id)
    pd.testing.assert_series_equal(base, after)


def test_missing_universe_still_builds(tables):
    out = F.build_features(**tables, companies=F.load_companies().iloc[0:0])
    assert out["mkt_ret_j0"].isna().all()
    assert (out["region_other"] == 1).all()


def test_attention_zscore_matches_dossier_window():
    v = pd.Series(np.r_[np.full(70, 100.0), 10_000.0])
    v.iloc[:70] += np.arange(70) % 3  # small noise so the std is not zero
    z = F.attention_zscore(v, window=60, gap=5)
    assert z.iloc[-1] > 50
    assert np.isnan(F.attention_zscore(v.iloc[:10], window=60, gap=5).iloc[-1])
