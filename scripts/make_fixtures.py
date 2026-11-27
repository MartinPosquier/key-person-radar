"""Generate small synthetic fixtures for every table of kpr.schemas (frozen file).

3 fake companies, 5 fake persons, trading days of 2020-2021. Each person gets 8 planted
attention shocks; on each shock the company's return receives an abnormal move of known
size (sensitivity x tone, with bad news weighted twice as much as good news). Downstream
sessions can therefore test that their code recovers the planted effects.

Run: python scripts/make_fixtures.py   (writes tests/fixtures/*.parquet)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kpr.utils.io import write_fixture

SEED = 42
DAYS = pd.bdate_range("2020-01-01", "2021-12-31")
COMPANIES = {  # company_id -> (market index, beta)
    "FAKE1": ("^GSPC", 1.3),
    "FAKE2": ("^GSPC", 0.9),
    "FAKE3": ("^STOXX", 1.0),
}
# person_id -> (company_id, role, founder, sensitivity, mandate start, mandate end)
PERSONS = {
    "p_alpha": ("FAKE1", "CEO", True, 3.0, "2005-01-01", None),
    "p_beta": ("FAKE1", "CFO", False, 0.8, "2015-06-01", None),
    "p_gamma": ("FAKE2", "CEO", False, 1.5, "2012-03-01", None),
    "p_delta": ("FAKE3", "CEO", False, 0.3, "2010-01-01", "2021-06-30"),
    "p_eps": ("FAKE3", "CEO", False, 1.0, "2021-07-01", None),
}
BIRTH_DATES = {  # one missing on purpose: real data will have gaps
    "p_alpha": "1963-02-17", "p_beta": "1975-09-01", "p_gamma": "1968-05-30",
    "p_delta": "1955-11-12", "p_eps": None,
}
SHOCKS_PER_PERSON = 8


def _active_days(start: str, end: str | None) -> pd.DatetimeIndex:
    end_ts = pd.Timestamp(end) if end else DAYS[-1]
    return DAYS[(DAYS >= pd.Timestamp(start)) & (DAYS <= end_ts)]


def main() -> None:
    rng = np.random.default_rng(SEED)
    n = len(DAYS)

    # Market, VIX ---------------------------------------------------------------------------
    market_rets = {idx: rng.normal(0.0003, 0.01, n) for idx in ("^GSPC", "^STOXX")}
    market = pd.concat(
        [
            pd.DataFrame(
                {"index_id": idx, "date": DAYS, "ret": r, "close": 100 * np.cumprod(1 + r)}
            )
            for idx, r in market_rets.items()
        ],
        ignore_index=True,
    )[["index_id", "date", "close", "ret"]]
    vix = pd.DataFrame({"date": DAYS, "vix": np.clip(rng.normal(20, 5, n), 9, 80)})

    # Shocks: dates and tone per person ---------------------------------------------------
    shocks = []
    for pid, (cid, _role, _founder, sens, start, end) in PERSONS.items():
        active = _active_days(start, end)
        candidates = active[30:-5]
        picks = np.sort(rng.choice(len(candidates), SHOCKS_PER_PERSON, replace=False))
        for k, i in enumerate(picks):
            tone = rng.choice([-1.0, 1.0]) * rng.uniform(1.2, 3.0)
            weight = 2.0 if tone < 0 else 1.0
            planted = sens * weight * np.sign(tone) * 0.01  # total abnormal move over 3 days
            shocks.append(
                {"person_id": pid, "company_id": cid, "date": candidates[i], "tone": tone,
                 "planted_car": planted, "k": k}
            )
    shocks = pd.DataFrame(shocks)

    # Prices with planted abnormal moves ----------------------------------------------------
    price_frames = []
    for cid, (idx, beta) in COMPANIES.items():
        ret = 0.0002 + beta * market_rets[idx] + rng.normal(0, 0.012, n)
        for _, s in shocks[shocks.company_id == cid].iterrows():
            i = DAYS.get_loc(s.date)
            ret[i : i + 3] += s.planted_car / 3
        close = 50 * np.cumprod(1 + ret)
        price_frames.append(pd.DataFrame({
            "company_id": cid, "date": DAYS, "close_adj": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
            "shares_out": 1e9, "ret": ret,
        }))
    prices = pd.concat(price_frames, ignore_index=True)

    # Attention: Wikipedia views and GDELT, with spikes on shock days ------------------------
    wiki_frames, gdelt_frames = [], []
    for pid in PERSONS:
        base = rng.lognormal(7, 0.3, n)
        n_art = rng.poisson(8, n)
        tone = rng.normal(0, 1, n)
        for _, s in shocks[shocks.person_id == pid].iterrows():
            i = DAYS.get_loc(s.date)
            base[i] *= rng.uniform(10, 30)
            n_art[i] = n_art[i] * 8 + 20
            tone[i] = s.tone * 2
        wiki_frames.append(pd.DataFrame({
            "page_id": pid, "date": DAYS, "views_en": base, "views_local": 0.0,
            "views_total": base,
        }))
        gdelt_frames.append(pd.DataFrame({
            "person_id": pid, "date": DAYS, "n_articles": n_art.astype("int64"),
            "tone_mean": tone, "tone_neg_share": np.clip(0.3 - tone / 10, 0, 1),
            "tone_std": np.abs(rng.normal(2, 0.5, n)),
        }))
    for cid in COMPANIES:
        views = rng.lognormal(8, 0.3, n)
        wiki_frames.append(pd.DataFrame({
            "page_id": f"company_{cid}", "date": DAYS, "views_en": views,
            "views_local": 0.0, "views_total": views,
        }))
    wiki_views = pd.concat(wiki_frames, ignore_index=True)
    gdelt_daily = pd.concat(gdelt_frames, ignore_index=True)

    trends_ratio = pd.DataFrame([
        {"person_id": pid, "year": y, "ratio_person_company": float(rng.uniform(0.05, 0.6))}
        for pid in PERSONS for y in (2020, 2021)
    ])

    earnings = pd.DataFrame([
        {"company_id": cid, "date": d, "quarter": f"Q{q}", "source": "fixture"}
        for cid in COMPANIES
        for d, q in zip(
            pd.to_datetime(["2020-02-05", "2020-05-06", "2020-08-05", "2020-11-04",
                            "2021-02-03", "2021-05-05", "2021-08-04", "2021-11-03"]),
            [4, 1, 2, 3, 4, 1, 2, 3], strict=True,
        )
    ])

    mandates = pd.DataFrame([
        {"person_id": pid, "company_id": cid, "role": role, "start": pd.Timestamp(start),
         "end": pd.Timestamp(end) if end else pd.NaT, "founder": founder,
         "birth_date": pd.Timestamp(BIRTH_DATES[pid]) if BIRTH_DATES[pid] else pd.NaT}
        for pid, (cid, role, founder, _s, start, end) in PERSONS.items()
    ])

    headlines = pd.DataFrame([
        {"person_id": s.person_id, "date": s.date, "headline_id": f"{s.person_id}-{s.k}-{j}",
         "title": f"{'Trouble' if s.tone < 0 else 'Praise'} for {s.person_id} ({j})",
         "source": "fixture"}
        for _, s in shocks.iterrows() for j in range(3)
    ])

    # Panel -------------------------------------------------------------------------------
    panel_frames = []
    earn_days = set(earnings.date)
    for pid, (cid, *_rest) in PERSONS.items():
        v = wiki_views[wiki_views.page_id == pid].set_index("date")["views_total"]
        g = gdelt_daily[gdelt_daily.person_id == pid].set_index("date")
        logv = np.log1p(v)
        ref = logv.shift(6).rolling(60, min_periods=20)
        z_wiki = (logv - ref.mean()) / ref.std()
        art = g["n_articles"].astype(float)
        refn = art.shift(6).rolling(60, min_periods=20)
        z_news = (art - refn.mean()) / refn.std()
        r = prices[prices.company_id == cid].set_index("date")["ret"]
        panel_frames.append(pd.DataFrame({
            "person_id": pid, "date": DAYS, "company_id": cid, "views": v.values,
            "z_wiki": z_wiki.values, "n_articles": g["n_articles"].values,
            "z_news": z_news.values, "tone_rel": g["tone_mean"].values, "ret": r.values,
            "earnings_window": [d in earn_days for d in DAYS], "vix": vix["vix"].values,
        }))
    panel = pd.concat(panel_frames, ignore_index=True)

    # Events (what the quant session will produce) ------------------------------------------
    def split_of(d: pd.Timestamp) -> str:
        if d.year == 2020:
            return "train"
        return "validation" if d.month <= 6 else "test"

    ev = shocks.copy()
    ev["event_id"] = ev.person_id + "-" + ev.date.dt.strftime("%Y%m%d")
    ev["car_0_2"] = ev.planted_car + rng.normal(0, 0.004, len(ev))
    events = pd.DataFrame({
        "event_id": ev.event_id, "person_id": ev.person_id, "company_id": ev.company_id,
        "date_utc": ev.date, "j0": ev.date, "z_wiki": rng.uniform(3, 8, len(ev)),
        "z_news": rng.uniform(2, 6, len(ev)), "n_articles": rng.integers(20, 120, len(ev)),
        "tone_rel": ev.tone, "tone_neg_share": np.clip(0.3 - ev.tone / 10, 0, 1),
        "sign": np.where(ev.tone < -1, "negative", np.where(ev.tone > 1, "positive", "neutral")),
        "car_0_2": ev.car_0_2, "car_m5_m1": rng.normal(0, 0.005, len(ev)),
        "car_0_1": ev.car_0_2 * 0.8, "car_0_5": ev.car_0_2 * 1.1,
        "excluded": False, "exclusion_reason": None, "is_natural_experiment": False,
        "split": [split_of(d) for d in ev.date],
    })
    events.loc[events.index[:2], ["is_natural_experiment", "split"]] = [True, "reserved"]
    events["exclusion_reason"] = events["exclusion_reason"].astype(object)

    headlines_tone = pd.DataFrame({
        "person_id": ev.person_id, "date": ev.date,
        "finbert_pos": np.where(ev.tone > 0, 0.7, 0.1),
        "finbert_neg": np.where(ev.tone < 0, 0.7, 0.1),
        "n_headlines": 3,
    })
    headlines_tone["finbert_net"] = headlines_tone.finbert_pos - headlines_tone.finbert_neg
    headlines_tone = headlines_tone[
        ["person_id", "date", "finbert_pos", "finbert_neg", "finbert_net", "n_headlines"]
    ]

    features = pd.DataFrame({
        "event_id": events.event_id, "z_wiki": events.z_wiki, "z_news": events.z_news,
        "tone_neg": np.maximum(0, -events.tone_rel), "tone_pos": np.maximum(0, events.tone_rel),
        "founder": events.person_id.map(lambda p: float(PERSONS[p][2])),
        "tenure_years": [
            (d - pd.Timestamp(PERSONS[p][4])).days / 365.25
            for p, d in zip(events.person_id, events.j0, strict=True)
        ],
        "ceo_age": [
            (d - pd.Timestamp(BIRTH_DATES[p])).days / 365.25 if BIRTH_DATES[p] else np.nan
            for p, d in zip(events.person_id, events.j0, strict=True)
        ],
    })

    score_rows = []
    for pid, (cid, _r, _f, sens, *_d) in PERSONS.items():
        for y in (2020, 2021):
            s_minus, s_plus = min(10.0, sens * 3), min(10.0, sens * 1.5)
            score_rows.append({
                "person_id": pid, "company_id": cid, "year": y,
                "s_minus_raw": sens * 2.0, "s_plus_raw": sens, "s_minus": s_minus,
                "s_plus": s_plus,
                "s_minus_lo": max(0.0, s_minus - 1), "s_minus_hi": min(10.0, s_minus + 1),
                "s_plus_lo": max(0.0, s_plus - 1), "s_plus_hi": min(10.0, s_plus + 1),
                "asymmetry": 2.0, "v_pct": sens * 2.0, "v_usd": sens * 1e9,
                "v_lo": sens * 1.0, "v_hi": sens * 3.0, "pi_share": min(1.0, sens / 10),
                "n_events": SHOCKS_PER_PERSON // 2, "method_agreement": 0.8,
                "succession_flag": pid == "p_eps" and y == 2021,
            })
    scores = pd.DataFrame(score_rows)

    tables = {
        "prices": prices, "market": market, "vix": vix, "wiki_views": wiki_views,
        "gdelt_daily": gdelt_daily, "trends_ratio": trends_ratio, "earnings": earnings,
        "mandates": mandates, "headlines": headlines, "panel": panel, "events": events,
        "headlines_tone": headlines_tone, "features": features, "scores": scores,
    }
    for name, df in tables.items():
        path = write_fixture(df.reset_index(drop=True), name)
        print(f"{name:<15} {len(df):>6} rows -> {path.name}")


if __name__ == "__main__":
    main()
