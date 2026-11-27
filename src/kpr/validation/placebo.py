"""Placebo tests.

Owner: quant session. Task Q8 in PARALLEL_SESSIONS.md §7. Dossier §11.3.

A placebo is a situation where nothing should happen. If an effect shows, the pipeline
measures noise or a bias.

- ``random_dates``: for each person, as many fake shock dates as real kept events, drawn
  among calm days. Expected: mean CAR zero; models cannot beat B0.
- ``unrelated_people``: attention shocks of unrelated celebrities (athletes, actors) put on
  the prices of our companies. Expected: no effect.
- ``wrong_company``: each real shock put on a company of another sector. Expected: no
  effect, or a weak one through the market.
- ``time_reversal``: CAR from J-10 to J-8, before the shock. Expected: no effect.

Placebo events keep the sign of a real event (random dates borrow the signs of the person's
real events, in random order) so their by-sign table reads like the real one. Every draw
uses a fixed seed.

``python -m kpr.validation.placebo`` runs the placebos on the events table and writes
`processed/validation/placebo_events.parquet` and `placebo_summary.csv` (real data only;
``--out-dir`` otherwise). `unrelated_people` runs only when a panel of unrelated persons is
given (``--unrelated-panel``, same columns as the panel table).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kpr.build import events as E
from kpr.build import returns as R
from kpr.utils.logging import get_logger
from kpr.validation import event_study as ES

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
SEED = 2026
PLACEBOS = ("random_dates", "unrelated_people", "wrong_company", "time_reversal")
# A calm day: ordinary attention and no shock of the person within this many days.
CALM_Z_MAX = 1.0
CALM_DISTANCE_DAYS = 10
OUT_COLUMNS = ["placebo", "event_id", "person_id", "company_id", "date_utc", "j0", "sign",
               "car_0_2", "n_est"]


def company_sectors(path: Path | None = None) -> dict[str, str]:
    """company_id -> sector from config/universe.yaml."""
    path = path or REPO_ROOT / "config" / "universe.yaml"
    with open(path, encoding="utf-8") as f:
        universe = yaml.safe_load(f) or {}
    return {c["company_id"]: c.get("sector", "unknown") for c in universe.get("companies") or []}


def _with_cars(placebo: str, ev: pd.DataFrame, markets: E.Markets, prices: pd.DataFrame,
               market: pd.DataFrame, params: Mapping,
               window: tuple[int, int] | None = None, recompute_j0: bool = True) -> pd.DataFrame:
    """J0 and CAR of placebo events (CAR[0, 2] unless another `window` is given)."""
    ev = ev.copy()
    if recompute_j0:
        ev["j0"] = [markets.j0(c, d) for c, d in zip(ev["company_id"], ev["date_utc"],
                                                       strict=True)]
        ev["j0"] = pd.to_datetime(ev["j0"])
    window = window or tuple(params["returns"]["car_window"])
    ok = ev["j0"].notna()
    mapping = {c: markets.index_of(c) for c in ev["company_id"].unique()}
    cars = R.compute_cars(ev.loc[ok, ["event_id", "company_id", "j0"]], prices, market, mapping,
                          windows={"car_0_2": window},
                          estimation_window=tuple(params["returns"]["estimation_window"]),
                          sessions_for=markets.sessions)
    ev = ev.drop(columns=["car_0_2", "n_est"], errors="ignore").merge(
        cars[["event_id", "car_0_2", "n_est"]], on="event_id", how="left")
    ev["n_est"] = ev["n_est"].fillna(0).astype(int)
    # Same estimation-history rule as the real events (dossier §7.4).
    ev.loc[ev["n_est"] < params["filters"]["min_estimation_obs"], "car_0_2"] = np.nan
    ev["placebo"] = placebo
    return ev[OUT_COLUMNS]


def calm_days(panel: pd.DataFrame, shocks: pd.DataFrame) -> pd.DataFrame:
    """Person-days with |z_wiki| and |z_news| below 1 and no shock within 10 days."""
    calm = panel[(panel["z_wiki"].abs() < CALM_Z_MAX) & (panel["z_news"].abs() < CALM_Z_MAX)]
    calm = calm[["person_id", "company_id", "date", "tone_rel"]].copy()
    near = pd.Series(False, index=calm.index)
    for pid, dates in shocks.groupby("person_id")["date_utc"]:
        rows = calm["person_id"] == pid
        d = calm.loc[rows, "date"].to_numpy()[:, None]
        dist = np.abs((d - dates.to_numpy()[None, :]) / np.timedelta64(1, "D"))
        near.loc[rows] = (dist <= CALM_DISTANCE_DAYS).any(axis=1)
    return calm[~near]


def random_dates(events: pd.DataFrame, panel: pd.DataFrame, rng: np.random.Generator,
                 all_shocks: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fake shocks: for each person, as many calm days as real kept events (same signs)."""
    shocks = events if all_shocks is None else all_shocks
    calm = calm_days(panel, shocks)
    rows = []
    for pid, real in events.groupby("person_id"):
        pool = calm[calm["person_id"] == pid]
        n = min(len(real), len(pool))
        if n < len(real):
            log.warning("%s: only %d calm days for %d events", pid, len(pool), len(real))
        if n == 0:
            continue
        pick = pool.iloc[np.sort(rng.choice(len(pool), n, replace=False))]
        rows.append(pd.DataFrame({
            "event_id": [f"{pid}-rd{i:03d}" for i in range(n)], "person_id": pid,
            "company_id": pick["company_id"].to_numpy(), "date_utc": pick["date"].to_numpy(),
            "sign": rng.permutation(real["sign"].to_numpy())[:n],
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["event_id", "person_id", "company_id", "date_utc", "sign"])


def wrong_company(events: pd.DataFrame, companies: list[str], sectors: Mapping[str, str],
                  rng: np.random.Generator) -> pd.DataFrame:
    """Each real shock put on a random company of another sector (any other if unknown)."""
    rows = []
    for ev in events.itertuples(index=False):
        own = sectors.get(ev.company_id)
        pool = [c for c in companies if c != ev.company_id
                and (own is None or sectors.get(c, "unknown") != own)]
        if not pool:
            continue
        rows.append({"event_id": f"{ev.event_id}-wc", "person_id": ev.person_id,
                     "company_id": pool[rng.integers(len(pool))], "date_utc": ev.date_utc,
                     "sign": ev.sign})
    return pd.DataFrame(rows, columns=["event_id", "person_id", "company_id", "date_utc",
                                       "sign"])


def unrelated_people(unrelated_panel: pd.DataFrame, companies: list[str],
                     rng: np.random.Generator, params: Mapping) -> pd.DataFrame:
    """Shocks of unrelated persons (§7.2 rule on their panel) put on random companies."""
    shocks = E.detect_shocks(unrelated_panel, params)
    shocks["sign"] = E.sign_of(shocks["tone_rel"], params["tone_sign_threshold"]).to_numpy()
    shocks["company_id"] = [companies[i] for i in rng.integers(len(companies), size=len(shocks))]
    shocks["event_id"] = (shocks["person_id"] + "-" + shocks["date_utc"].dt.strftime("%Y%m%d")
                          + "-up")
    return shocks[["event_id", "person_id", "company_id", "date_utc", "sign"]]


def run_placebos(events: pd.DataFrame, panel: pd.DataFrame, prices: pd.DataFrame,
                 market: pd.DataFrame, *, unrelated_panel: pd.DataFrame | None = None,
                 universe=None, sectors: Mapping[str, str] | None = None,
                 params: Mapping | None = None, seed: int = SEED) -> pd.DataFrame:
    """All placebos on the kept, non-test events; one row per placebo event."""
    params = params or E.load_params()
    sectors = company_sectors() if sectors is None else sectors
    rng = np.random.default_rng(seed)
    base = ES.select(events)
    markets = E.Markets(prices, market, universe)
    companies = sorted(prices["company_id"].unique())

    out = [
        _with_cars("random_dates", random_dates(base, panel, rng, all_shocks=events),
                   markets, prices, market, params),
        _with_cars("wrong_company", wrong_company(base, companies, sectors, rng),
                   markets, prices, market, params),
        _with_cars("time_reversal", base.assign(event_id=base["event_id"] + "-tr"),
                   markets, prices, market, params, window=(-10, -8), recompute_j0=False),
    ]
    if unrelated_panel is not None:
        out.append(_with_cars("unrelated_people",
                              unrelated_people(unrelated_panel, companies, rng, params),
                              markets, prices, market, params))
    return pd.concat(out, ignore_index=True)


def summary(placebo_events: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Mean CAR tests by placebo and sign, with the real events first for comparison."""
    tables = []
    if events is not None:
        real = ES.mean_car_table(ES.select(events)).assign(placebo="real")
        tables.append(real)
    for name, g in placebo_events.groupby("placebo", sort=False):
        tables.append(ES.mean_car_table(g).assign(placebo=name))
    return pd.concat(tables).reset_index().set_index(["placebo", "sign"])


def main(argv: list[str] | None = None) -> int:
    from kpr.utils.io import data_dir, read_table, use_fixtures

    parser = argparse.ArgumentParser(prog="python -m kpr.validation.placebo",
                                     description="Placebo tests of dossier §11.3.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--unrelated-panel", type=Path,
                        help="parquet panel of unrelated persons (athletes, actors)")
    parser.add_argument("--out-dir", type=Path, help="default: processed/validation/")
    args = parser.parse_args(argv)

    events, panel = read_table("events"), read_table("panel")
    prices, market = read_table("prices"), read_table("market")
    unrelated = pd.read_parquet(args.unrelated_panel) if args.unrelated_panel else None
    placebo_events = run_placebos(events, panel, prices, market, unrelated_panel=unrelated,
                                  seed=args.seed)
    table = summary(placebo_events, events)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table[["n", "mean", "p_t", "p_sign"]].to_string())

    out_dir = args.out_dir
    if out_dir is None and not use_fixtures():
        out_dir = data_dir() / "processed" / "validation"
    if out_dir is None:
        log.info("fixtures: nothing written (pass --out-dir)")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    placebo_events.to_parquet(out_dir / "placebo_events.parquet", index=False)
    table.to_csv(out_dir / "placebo_summary.csv")
    log.info("wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
