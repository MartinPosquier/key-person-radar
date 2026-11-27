"""Feature matrix (5 families), no look-ahead.

Owner: ml session. Task M2 in PARALLEL_SESSIONS.md §7. Dossier §9.2.

One row per event of the `events` table. Every feature is known at the latest on J0:
attention and tone come from the shock day (date_utc <= j0), price-based context from the
sessions strictly before J0 (except the market return and VIX of J0, listed in §9.2), and
yearly attention ratios from the previous calendar year.

Families (§9.2):
  A intensity    z_wiki, z_news, log_views, log_articles, peak_days, local_en_ratio
  B tone         tone_rel, tone_neg_share, tone_std, finbert_net, tone_neg, tone_pos
  C person       role_*, founder, ceo_age, tenure_years, attention_share, views_ratio
  D specificity  z_company, specificity
  E market       log_mcap, vol_20d, car_m5_m1, vix, mkt_ret_j0, region_*, sector_*
  interactions   tone_x_attention, tone_x_founder

Scaling is NOT done here: it belongs inside each model's scikit-learn pipeline, fitted on
the training fold only (§9.2, LAB 4 Q5). Categorical variables are one-hot encoded on fixed
category lists, which uses no data and therefore cannot leak.

Run: python -m kpr.build.features   (KPR_USE_FIXTURES=1 to read the fixtures)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kpr.utils.io import read_table, write_table
from kpr.utils.logging import get_logger

log = get_logger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
DAY = np.timedelta64(1, "D")

FAMILIES = {
    "A_intensity": ["z_wiki", "z_news", "log_views", "log_articles", "peak_days",
                    "local_en_ratio"],
    "B_tone": ["tone_rel", "tone_neg_share", "tone_std", "finbert_net", "tone_neg",
               "tone_pos"],
    "C_person": ["role_*", "founder", "ceo_age", "tenure_years", "attention_share",
                 "views_ratio"],
    "D_specificity": ["z_company", "specificity"],
    "E_market": ["log_mcap", "vol_20d", "car_m5_m1", "vix", "mkt_ret_j0", "region_*",
                 "sector_*"],
    "interactions": ["tone_x_attention", "tone_x_founder"],
}


# ---- configuration --------------------------------------------------------------------------


def load_params() -> dict:
    """Feature parameters (config/params/ml.yaml) plus the attention window of quant.yaml."""
    ml = yaml.safe_load((CONFIG_DIR / "params" / "ml.yaml").read_text(encoding="utf-8"))
    quant = yaml.safe_load((CONFIG_DIR / "params" / "quant.yaml").read_text(encoding="utf-8"))
    params = dict(ml["features"])
    params["attention_window"] = quant["attention"]["window_days"]
    params["attention_gap"] = quant["attention"]["gap_days"]
    return params


def load_companies() -> pd.DataFrame:
    """company_id, region, sector, market_index from config/universe.yaml (may be empty)."""
    raw = yaml.safe_load((CONFIG_DIR / "universe.yaml").read_text(encoding="utf-8")) or {}
    cols = ["company_id", "region", "sector", "market_index"]
    return pd.DataFrame(raw.get("companies") or [], columns=cols)[cols]


def person_page_id(person_id: str) -> str:
    """page_id of a person's Wikipedia page in `wiki_views` (convention of the fixtures)."""
    return person_id


def company_page_id(company_id: str) -> str:
    """page_id of a company's Wikipedia page in `wiki_views` (convention of the fixtures)."""
    return f"company_{company_id}"


def family_columns(columns, family: str) -> list[str]:
    """Columns of `columns` that belong to a family of FAMILIES (expands the `prefix_*`)."""
    out = []
    for pattern in FAMILIES[family]:
        if pattern.endswith("*"):
            out += [c for c in columns if c.startswith(pattern[:-1])]
        elif pattern in columns:
            out.append(pattern)
    return out


# ---- helpers --------------------------------------------------------------------------------


def attention_zscore(views: pd.Series, window: int, gap: int) -> pd.Series:
    """z-score of log(1 + views) against the `window` days ending `gap + 1` days before (§7.1)."""
    x = np.log1p(views.astype(float))
    ref = x.shift(gap + 1).rolling(window, min_periods=max(2, window // 3))
    return (x - ref.mean()) / ref.std()


def _series(df: pd.DataFrame, key: str, value, column: str) -> pd.Series:
    out = df.loc[df[key] == value, ["date", column]]
    return out.drop_duplicates("date").set_index("date")[column].sort_index()


def _value_at(series: pd.Series, date) -> float:
    return float(series.get(date, np.nan)) if len(series) else np.nan


def _before(series: pd.Series, date, n: int) -> pd.Series:
    """The last `n` values strictly before `date`."""
    return series[series.index < date].iloc[-n:]


def _window_mean(series: pd.Series, end, days: int) -> float:
    """Mean over the `days` calendar days strictly before `end`."""
    sel = series[(series.index >= end - days * DAY) & (series.index < end)]
    return float(sel.mean()) if len(sel) else np.nan


def _peak_days(z: pd.Series, date, threshold: float) -> int:
    """Consecutive days up to and including `date` with z above `threshold` (no future days)."""
    run = 0
    for value in z[z.index <= date].iloc[::-1]:
        if not value > threshold:
            break
        run += 1
    return run


def _one_hot(values: pd.Series, prefix: str, categories: list[str]) -> pd.DataFrame:
    cats = list(dict.fromkeys([*categories, "other"]))
    cleaned = values.fillna("other").astype(str)
    cleaned = cleaned.where(cleaned.isin(cats), "other")
    return pd.DataFrame(
        {f"{prefix}_{c}": (cleaned == c).astype(float) for c in cats}, index=values.index
    )


def _active_mandate(mandates: pd.DataFrame, person_id: str, company_id: str, date) -> dict:
    m = mandates[(mandates["person_id"] == person_id) & (mandates["company_id"] == company_id)]
    active = m[(m["start"] <= date) & (m["end"].isna() | (m["end"] >= date))]
    pick = active if len(active) else m
    if pick.empty:
        return {"role": "other", "founder": False, "start": pd.NaT, "birth_date": pd.NaT}
    return pick.sort_values("start").iloc[-1].to_dict()


def _years_between(start, end) -> float:
    """Years from `start` to `end` (NaN when `start` is unknown): age, tenure."""
    return (end - start).days / 365.25 if pd.notna(start) else np.nan


# ---- main computation -----------------------------------------------------------------------


def build_features(
    events: pd.DataFrame,
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    market: pd.DataFrame,
    vix: pd.DataFrame,
    mandates: pd.DataFrame,
    trends_ratio: pd.DataFrame,
    wiki_views: pd.DataFrame,
    gdelt_daily: pd.DataFrame,
    headlines_tone: pd.DataFrame | None = None,
    companies: pd.DataFrame | None = None,
    params: dict | None = None,
) -> pd.DataFrame:
    """Return the `features` table: `event_id` plus one numeric column per feature."""
    p = params or load_params()
    companies = companies if companies is not None else load_companies()
    meta = companies.drop_duplicates("company_id").set_index("company_id")
    vix_s = vix.drop_duplicates("date").set_index("date")["vix"].sort_index()
    trends = trends_ratio.set_index(["person_id", "year"])["ratio_person_company"]
    has_finbert = headlines_tone is not None and len(headlines_tone) > 0
    finbert = headlines_tone.set_index(["person_id", "date"])["finbert_net"] if has_finbert \
        else pd.Series(dtype=float)
    prices = prices.assign(mcap=prices["close_adj"] * prices["shares_out"])

    rows = []
    for e in events.itertuples(index=False):
        d, j0, pid, cid = e.date_utc, e.j0, e.person_id, e.company_id
        views = _series(panel, "person_id", pid, "views")
        z_person = _series(panel, "person_id", pid, "z_wiki")
        en = _value_at(_series(wiki_views, "page_id", person_page_id(pid), "views_en"), d)
        local = _value_at(_series(wiki_views, "page_id", person_page_id(pid), "views_local"), d)
        company_views = _series(wiki_views, "page_id", company_page_id(cid), "views_total")
        z_company = attention_zscore(company_views, p["attention_window"], p["attention_gap"])
        ret = _series(prices, "company_id", cid, "ret")
        cap = _series(prices, "company_id", cid, "mcap").dropna()
        index_id = meta["market_index"].get(cid)
        mkt = _series(market, "index_id", index_id, "ret")
        mandate = _active_mandate(mandates, pid, cid, j0)
        last_cap = _before(cap, j0, 1)
        vix_known = vix_s[vix_s.index <= j0]
        company_mean = _window_mean(company_views, d, p["ratio_window_days"])
        zc = _value_at(z_company, d)

        rows.append({
            "event_id": e.event_id,
            # A. intensity
            "z_wiki": e.z_wiki,
            "z_news": e.z_news,
            "log_views": np.log1p(_value_at(views, d)),
            "log_articles": np.log1p(e.n_articles),
            "peak_days": float(_peak_days(z_person, d, p["peak_z"])),
            "local_en_ratio": local / en if en > 0 else np.nan,
            # B. tone
            "tone_rel": e.tone_rel,
            "tone_neg_share": e.tone_neg_share,
            "tone_std": _value_at(_series(gdelt_daily, "person_id", pid, "tone_std"), d),
            "finbert_net": float(finbert.get((pid, d), np.nan)),
            # C. person
            "role": mandate["role"],
            "founder": float(bool(mandate["founder"])),
            "ceo_age": _years_between(mandate.get("birth_date"), j0),
            "tenure_years": _years_between(mandate["start"], j0),
            "attention_share": float(trends.get((pid, j0.year - 1), np.nan)),
            "views_ratio": _window_mean(views, d, p["ratio_window_days"]) / company_mean
            if company_mean > 0 else np.nan,
            # D. specificity: does the shock really concern the person?
            "z_company": zc,
            "specificity": e.z_wiki - zc,
            # E. market context
            "log_mcap": np.log(last_cap.iloc[0]) if len(last_cap) else np.nan,
            "vol_20d": _before(ret, j0, p["vol_window"]).std(),
            "car_m5_m1": e.car_m5_m1,
            "vix": float(vix_known.iloc[-1]) if len(vix_known) else np.nan,
            "mkt_ret_j0": _value_at(mkt, j0),
            "region": meta["region"].get(cid),
            "sector": meta["sector"].get(cid),
        })

    out = pd.DataFrame(rows)
    out["tone_neg"] = np.maximum(0.0, -out["tone_rel"])
    out["tone_pos"] = np.maximum(0.0, out["tone_rel"])
    out["tone_x_attention"] = out["tone_rel"] * out["attention_share"]
    out["tone_x_founder"] = out["tone_rel"] * out["founder"]

    sectors = sorted(companies["sector"].dropna().unique().tolist())
    dummies = [
        _one_hot(out.pop("role"), "role", p["roles"]),
        _one_hot(out.pop("region"), "region", p["regions"]),
        _one_hot(out.pop("sector"), "sector", sectors),
    ]
    out = pd.concat([out, *dummies], axis=1)
    numeric = out.columns.drop("event_id")
    out[numeric] = out[numeric].astype(float)
    return out


def main(argv: list[str] | None = None) -> int:
    tables = {name: read_table(name) for name in (
        "events", "panel", "prices", "market", "vix", "mandates", "trends_ratio",
        "wiki_views", "gdelt_daily",
    )}
    try:
        tables["headlines_tone"] = read_table("headlines_tone")
    except FileNotFoundError:
        log.info("headlines_tone not available yet: finbert_net left empty (task M5)")
    features = build_features(**tables)
    missing = features.isna().mean().sort_values(ascending=False)
    log.info("missing share per feature (top 5):\n%s", missing.head().to_string())
    path = write_table(features, "features")
    log.info("features: %d events x %d columns -> %s", *features.shape, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
