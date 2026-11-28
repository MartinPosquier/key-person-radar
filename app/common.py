"""Shared helpers for the Streamlit pages: data loading, honest-display rules, formatting.

The app never computes anything heavy (dossier §12): it reads precomputed files from
`app/data/`. Those files are written by `python -m kpr.scores.export_app` (real results,
task M13) or by `python scripts/ml_make_app_demo.py` (synthetic demo built from the test
fixtures, task M1).

Owner: ml session.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

# KPR_APP_DATA points the app to another export folder (tests, previews).
DATA_DIR = Path(os.environ.get("KPR_APP_DATA") or Path(__file__).resolve().parent / "data")

# Honest-display rules, dossier §12.4 and §10.6.
LOW_CONFIDENCE_MIN_EVENTS = 8
DISCLAIMER = "Research project. Not investment advice."
SCORE_CAP = 10.0

# Files the app reads, with the columns each page relies on. Extra columns are allowed.
APP_FILES: dict[str, tuple[str, ...]] = {
    # One row per (person, company, year): the `scores` table of kpr.schemas.
    "scores": (
        "person_id", "company_id", "year", "s_minus_raw", "s_plus_raw", "s_minus", "s_plus",
        "s_minus_lo", "s_minus_hi", "s_plus_lo", "s_plus_hi", "asymmetry", "v_pct", "v_usd",
        "v_lo", "v_hi", "pi_share", "n_events", "method_agreement", "succession_flag",
    ),
    # One row per shock: the `events` table, plus `car_pred` (model prediction, may be NaN).
    "events": (
        "event_id", "person_id", "company_id", "j0", "z_wiki", "tone_rel", "sign",
        "car_0_2", "car_pred", "excluded", "split",
    ),
    # Daily series restricted to tracked persons: attention and abnormal return.
    "daily": ("person_id", "date", "company_id", "views", "close_adj", "ar"),
    # Display names and filters (from config/persons.yaml and config/universe.yaml).
    "entities": (
        "person_id", "person_name", "company_id", "company_name", "role", "founder",
        "region", "sector", "group",
    ),
}


class AppDataError(RuntimeError):
    """Raised when a file of app/data/ is missing or lacks a required column."""


# ---- loading ------------------------------------------------------------------------------


def load_frame(name: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Read app/data/<name>.parquet and check it has the columns the pages need."""
    if name not in APP_FILES:
        raise KeyError(f"unknown app file {name!r}; known: {sorted(APP_FILES)}")
    path = Path(data_dir or DATA_DIR) / f"{name}.parquet"
    if not path.exists():
        raise AppDataError(
            f"{path} is missing. Run `python scripts/ml_make_app_demo.py` (demo data) "
            "or `python -m kpr.scores.export_app` (real results)."
        )
    df = pd.read_parquet(path)
    missing = [c for c in APP_FILES[name] if c not in df.columns]
    if missing:
        raise AppDataError(f"{path.name} lacks columns {missing}")
    return df


def load_meta(data_dir: Path | None = None) -> dict:
    """Read app/data/meta.json (data origin, export date). Empty dict if absent."""
    path = Path(data_dir or DATA_DIR) / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_case_study(data_dir: Path | None = None) -> dict | None:
    """Read app/data/case_study.json (Nvidia 2026 case, task Q12), or None if absent."""
    path = Path(data_dir or DATA_DIR) / "case_study.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def scores_with_names(scores: pd.DataFrame, entities: pd.DataFrame) -> pd.DataFrame:
    """Join display names and filter columns onto the scores table."""
    ent = entities.drop_duplicates(["person_id", "company_id"])
    out = scores.merge(ent, on=["person_id", "company_id"], how="left")
    out["person_name"] = out["person_name"].fillna(out["person_id"])
    out["company_name"] = out["company_name"].fillna(out["company_id"])
    out["low_confidence"] = out["n_events"] < LOW_CONFIDENCE_MIN_EVENTS
    return out


def latest_year(scores: pd.DataFrame) -> pd.DataFrame:
    """Keep, for each (person, company), the row of its most recent year."""
    idx = scores.groupby(["person_id", "company_id"])["year"].idxmax()
    return scores.loc[idx].reset_index(drop=True)


def company_dependence(scores: pd.DataFrame) -> pd.DataFrame:
    """Company score of dossier §3.7, per (company, year).

    dependence index = max of V over tracked persons (the most critical person);
    upper bound = sum of V (may count one event twice, hence shown only as a bound).
    """
    grouped = scores.groupby(["company_id", "year"])
    out = grouped["v_pct"].agg(dependence_index="max", upper_bound="sum").reset_index()
    out["n_persons"] = grouped["person_id"].nunique().values
    return out


def mean_car_curve(
    events: pd.DataFrame, daily: pd.DataFrame, window: int = 10
) -> pd.DataFrame:
    """Mean cumulative abnormal return from J-window to J+window, by shock sign.

    Uses the daily abnormal returns of app/data/daily.parquet. The cumulation starts at
    J-window, so the curve shows both anticipation and reaction (dossier §7.6).
    """
    rows = []
    for company_id, ev in events.groupby("company_id"):
        series = (
            daily[daily["company_id"] == company_id]
            .drop_duplicates("date")
            .set_index("date")["ar"]
            .sort_index()
        )
        dates = series.index
        for _, e in ev.iterrows():
            pos = dates.searchsorted(e["j0"])
            if pos >= len(dates) or pos - window < 0 or pos + window >= len(dates):
                continue
            ar = series.iloc[pos - window : pos + window + 1].to_numpy()
            for k, car in zip(range(-window, window + 1), ar.cumsum(), strict=True):
                rows.append({"event_id": e["event_id"], "sign": e["sign"], "k": k, "car": car})
    if not rows:
        return pd.DataFrame(columns=["sign", "k", "car", "n"])
    curves = pd.DataFrame(rows)
    return (
        curves.groupby(["sign", "k"])
        .agg(car=("car", "mean"), n=("event_id", "nunique"))
        .reset_index()
    )


# ---- formatting (honest display, dossier §12.4) -------------------------------------------


def fmt_score(value: float) -> str:
    """0-10 score with one decimal; values at the cap are flagged (dossier §3.3)."""
    if pd.isna(value):
        return "n/a"
    return f"{value:.1f}" + (" (capped)" if value >= SCORE_CAP else "")


def fmt_v(v_pct: float, v_lo: float, v_hi: float) -> str:
    """V always comes with its 5th-95th percentile interval and the word 'estimate'."""
    if pd.isna(v_pct):
        return "n/a"
    interval = "" if pd.isna(v_lo) or pd.isna(v_hi) else f" [{v_lo:.1f}–{v_hi:.1f}%]"
    return f"{v_pct:.1f}%{interval} (estimate)"


def fmt_usd(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    for unit, size in (("tn", 1e12), ("bn", 1e9), ("m", 1e6)):
        if abs(value) >= size:
            return f"${value / size:.1f}{unit}"
    return f"${value:,.0f}"


def confidence_badge(n_events: int) -> str:
    """Badge text for scores built on fewer than 8 shocks, else an empty string."""
    return "low confidence" if n_events < LOW_CONFIDENCE_MIN_EVENTS else ""


# ---- Streamlit page chrome ----------------------------------------------------------------


def page_header(title: str, subtitle: str | None = None) -> None:
    """Title, the mandatory disclaimer, and a banner when the data is a synthetic demo."""
    import streamlit as st

    st.title(title)
    if subtitle:
        st.caption(subtitle)
    st.warning(DISCLAIMER, icon="⚠️")
    meta = cached_meta()
    if meta.get("demo"):
        st.info(
            "Demo mode: the figures below come from synthetic data (fake persons and "
            "companies) used to build the app. Real estimates will replace them.",
            icon="🧪",
        )


def show_load_error(err: Exception) -> None:
    import streamlit as st

    st.error(f"App data could not be loaded: {err}")
    st.stop()


def _cache(func):
    import streamlit as st

    return st.cache_data(show_spinner=False)(func)


@_cache
def cached_frame(name: str) -> pd.DataFrame:
    return load_frame(name)


@_cache
def cached_meta() -> dict:
    return load_meta()


@_cache
def cached_case_study() -> dict | None:
    return load_case_study()


def load_all() -> dict[str, pd.DataFrame]:
    """All app files, cached; stops the page with a readable error if one is missing."""
    try:
        frames = {name: cached_frame(name) for name in APP_FILES}
    except AppDataError as err:
        show_load_error(err)
    frames["scores_named"] = scores_with_names(frames["scores"], frames["entities"])
    return frames
