"""Nvidia 2026 out-of-period case study.

Owner: quant session. Task Q12 in PARALLEL_SESSIONS.md §7. Dossier §11.6.

A single case proves nothing statistically: it is an illustration (the hook of the defence
and of the app). Steps of §11.6:

1. Source the event: the attention peak is read in Jensen Huang's 2026 Wikipedia views
   (``find_attention_peaks``); a date sourced by hand overrides it (``--date``).
2. Real CAR of Nvidia on [0, 2] and [0, 5], net of the market, next to the raw return; the
   same windows for sector peers (if the whole sector fell, Huang's own effect is smaller).
3. Competing news the same days: earnings releases within the window and the person's
   headlines (``competing_news``); export restrictions, customer news and sector moves
   still need a manual check, listed in the report.
4. Model comparison: Huang's S-, S+ and V at the end of 2025 and the reaction they imply
   for a shock of this sign (``model_expectation``).
5. Honest verdict: consistent, underestimated or overestimated (``verdict``).

The case lies outside the 2014-2025 study period, so its data (prices and pageviews of
2026) are read from separate files given on the command line, never from the shared
study tables, so that no 2026 row can reach the events or the models.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from kpr.build import events as E
from kpr.build import returns as R
from kpr.utils.logging import get_logger

log = get_logger(__name__)

PERSON, COMPANY, YEAR = "jensen_huang", "NVDA", 2026
PEERS = ("AMD", "INTC", "ASML")  # semiconductors of the universe
WINDOWS = {"car_0_2": (0, 2), "car_0_5": (0, 5)}
# Ratio actual / expected magnitude inside this band reads as "consistent".
CONSISTENT_BAND = (0.5, 2.0)


def find_attention_peaks(views: pd.DataFrame, page_id: str, year: int, top: int = 3,
                         params: Mapping | None = None) -> pd.DataFrame:
    """The `top` days of `year` with the highest attention z-score of the page (§7.1)."""
    att = (params or E.load_params())["attention"]
    s = views[views["page_id"] == page_id].set_index("date")["views_total"].sort_index()
    if s.empty:
        raise ValueError(f"no views for page {page_id!r}")
    z = E.attention_z(s, att["window_days"], att["gap_days"]).rename("z_wiki")
    out = pd.concat([s.rename("views"), z], axis=1)
    out = out[out.index.year == year].dropna(subset=["z_wiki"])
    return out.sort_values("z_wiki", ascending=False).head(top)


def case_cars(prices: pd.DataFrame, market: pd.DataFrame, companies: Sequence[str],
              day: pd.Timestamp, universe=None, params: Mapping | None = None) -> pd.DataFrame:
    """CAR on [0, 2] and [0, 5] (market model) plus raw and market returns, per company."""
    params = params or E.load_params()
    markets = E.Markets(prices, market, universe)
    stock = {c: g.set_index("date")["ret"].sort_index() for c, g in prices.groupby("company_id")}
    rows = []
    for cid in companies:
        if cid not in stock:
            log.warning("no prices for %s", cid)
            continue
        j0 = markets.j0(cid, day)
        if pd.isna(j0):
            continue
        res = R.event_abnormal_returns(
            stock[cid], markets.index_ret(markets.index_of(cid)), j0,
            sessions=markets.sessions(cid),
            estimation_window=tuple(params["returns"]["estimation_window"]),
            event_window=(-10, 10))
        f = res.frame
        row = {"company_id": cid, "j0": j0, "beta": res.fit.params.get("beta", np.nan),
               "n_est": res.fit.n_obs}
        for name, (lo, hi) in WINDOWS.items():
            days = f.loc[lo:hi]
            row[name] = res.car(lo, hi)
            row[f"raw_{name[4:]}"] = days["ret"].sum() if days["ret"].notna().all() else np.nan
            row[f"market_{name[4:]}"] = days["ret_m"].sum()
        rows.append(row)
    return pd.DataFrame(rows).set_index("company_id")


def competing_news(day: pd.Timestamp, company: str, person: str, earnings: pd.DataFrame,
                   headlines: pd.DataFrame | None = None, days: int = 5) -> dict:
    """Earnings releases within `days` days and the person's headlines of the day."""
    near = earnings[(earnings["company_id"] == company)
                    & ((earnings["date"] - day).dt.days.abs() <= days)]
    out = {"earnings_dates": [d.date().isoformat() for d in near["date"]], "headlines": []}
    if headlines is not None and len(headlines):
        h = headlines[(headlines["person_id"] == person)
                      & ((headlines["date"] - day).dt.days.abs() <= 1)]
        out["headlines"] = h["title"].head(20).tolist()
    return out


def model_expectation(scores: pd.DataFrame, person: str, company: str, year: int,
                      sign: str = "negative") -> dict:
    """End-of-`year` scores and the CAR magnitude they imply for a shock of `sign`.

    s_minus_raw / s_plus_raw are sensitivities "in % of CAR" (§10.6): the expected
    magnitude of the reaction to a typical negative / positive shock.
    """
    row = scores[(scores["person_id"] == person) & (scores["company_id"] == company)
                 & (scores["year"] == year)]
    if row.empty:
        raise ValueError(f"no score for {person} at {company} in {year}")
    r = row.iloc[0]
    raw = r["s_minus_raw"] if sign == "negative" else r["s_plus_raw"]
    keep = ["s_minus", "s_plus", "s_minus_lo", "s_minus_hi", "s_plus_lo", "s_plus_hi",
            "v_pct", "v_lo", "v_hi", "s_minus_raw", "s_plus_raw", "n_events"]
    out = {k: (float(r[k]) if pd.notna(r[k]) else None) for k in keep}
    out.update(year=year, sign=sign, expected_abs_car=float(raw) / 100.0)
    return out


def verdict(actual_car: float, expected_abs_car: float,
            band: tuple[float, float] = CONSISTENT_BAND) -> str:
    """consistent / underestimated / overestimated from |actual| / expected."""
    if not np.isfinite(actual_car) or not expected_abs_car:
        return "not computable"
    ratio = abs(actual_car) / expected_abs_car
    if ratio < band[0]:
        return f"overestimated (actual is {ratio:.1f}x the expected magnitude)"
    if ratio > band[1]:
        return f"underestimated (actual is {ratio:.1f}x the expected magnitude)"
    return f"consistent (actual is {ratio:.1f}x the expected magnitude)"


def _markdown_table(df: pd.DataFrame) -> str:
    """Small Markdown table (no extra dependency); floats with 4 decimals."""
    def cell(v):
        if isinstance(v, float):
            if np.isnan(v):
                return ""
            return f"{int(v)}" if v.is_integer() and abs(v) >= 1 else f"{v:.4f}"
        return str(v)

    header = ["", *map(str, df.columns)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx), *map(cell, row.tolist())]) + " |")
    return "\n".join(lines)


def report(case: Mapping) -> str:
    """Markdown summary of the case for the report and the app."""
    cars = pd.DataFrame(case["cars"]).T
    lines = [
        f"# Case study: {case['company']} / {case['person']} ({case['date']})",
        "",
        f"Date source: {case['date_source']}. J0 = {case['j0']}.",
        "",
        "## Returns (net of the market)",
        "",
        _markdown_table(cars),
        "",
        "## Competing news",
        "",
        f"- Earnings within 5 days: {case['news']['earnings_dates'] or 'none'}",
        f"- Headlines of the day: {len(case['news']['headlines'])}",
        "- To check by hand: export restrictions, customer announcements, sector moves.",
        "",
        "## Model",
        "",
        f"- Scores at the end of {case['model']['year']}: S- {case['model']['s_minus']}, "
        f"S+ {case['model']['s_plus']}, V {case['model']['v_pct']} %",
        f"- Expected |CAR| for a {case['model']['sign']} shock: "
        f"{case['model']['expected_abs_car']:.2%}",
        f"- Verdict: **{case['verdict']}**",
        "",
        "A single case is an illustration, not a statistical test (dossier §11.6).",
    ]
    return "\n".join(lines)


def run_case(views: pd.DataFrame, prices: pd.DataFrame, market: pd.DataFrame,
             scores: pd.DataFrame, earnings: pd.DataFrame, *, person: str = PERSON,
             company: str = COMPANY, peers: Sequence[str] = PEERS, year: int = YEAR,
             date: str | None = None, sign: str = "negative",
             headlines: pd.DataFrame | None = None, universe=None) -> dict:
    """All steps of §11.6; returns a JSON-ready dict."""
    if date:
        day, source = pd.Timestamp(date), "sourced by hand (--date)"
    else:
        peaks = find_attention_peaks(views, person, year)
        day = peaks.index[0]
        source = f"highest attention z-score of {year} (z = {peaks['z_wiki'].iloc[0]:.1f})"
    cars = case_cars(prices, market, [company, *peers], day, universe)
    if company not in cars.index:
        raise ValueError(f"no CAR for {company} around {day.date()}")
    model = model_expectation(scores, person, company, year - 1, sign)
    return {
        "person": person, "company": company, "date": day.date().isoformat(),
        "date_source": source, "j0": cars.loc[company, "j0"].date().isoformat(),
        "cars": json.loads(cars.drop(columns="j0").to_json(orient="index")),
        "news": competing_news(day, company, person, earnings, headlines),
        "model": model,
        "verdict": verdict(cars.loc[company, "car_0_2"], model["expected_abs_car"]),
    }


def main(argv: list[str] | None = None) -> int:
    from kpr.utils.io import data_dir, read_table, use_fixtures

    parser = argparse.ArgumentParser(prog="python -m kpr.validation.case_nvidia",
                                     description="Nvidia 2026 case study (dossier §11.6).")
    parser.add_argument("--views", type=Path, help="2026 pageviews parquet (wiki_views columns)")
    parser.add_argument("--prices", type=Path, help="2025-2026 prices parquet (prices columns)")
    parser.add_argument("--market", type=Path, help="2025-2026 index parquet (market columns)")
    parser.add_argument("--date", help="event day sourced by hand (YYYY-MM-DD)")
    parser.add_argument("--person", default=PERSON)
    parser.add_argument("--company", default=COMPANY)
    parser.add_argument("--peers", nargs="*", default=list(PEERS))
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--sign", choices=["negative", "positive"], default="negative")
    parser.add_argument("--out-dir", type=Path, help="default: processed/validation/")
    args = parser.parse_args(argv)

    def table(path, name):
        return pd.read_parquet(path) if path else read_table(name)

    case = run_case(table(args.views, "wiki_views"), table(args.prices, "prices"),
                    table(args.market, "market"), read_table("scores"),
                    read_table("earnings"), person=args.person, company=args.company,
                    peers=args.peers, year=args.year, date=args.date, sign=args.sign)
    text = report(case)
    print(text)

    out_dir = args.out_dir
    if out_dir is None and not use_fixtures():
        out_dir = data_dir() / "processed" / "validation"
    if out_dir is None:
        log.info("fixtures: nothing written (pass --out-dir)")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "case_nvidia.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
    (out_dir / "case_nvidia.md").write_text(text + "\n", encoding="utf-8")
    log.info("wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
