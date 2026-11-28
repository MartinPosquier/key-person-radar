"""Lambda calibration and implied value share V.

Owner: ml session. Task M10 in PARALLEL_SESSIONS.md §7. Dossier §3.5, §3.6, §10.4.

1. Calibration departures: abrupt departures NOT wanted by the company (death, illness,
   arrest, personal scandal) from config/natural_experiments.csv, dated up to the end of
   the training period. Later ones stay for the validation of task Q10 (§10.4: "on sépare
   la liste en deux, par date"). Each is matched to its event in the events table.
2. lambda = mean |actual CAR| / mean |predicted CAR| over those departures (§3.5).
3. Departure shock: the mean z_wiki, tone and specificity of the calibration departures.
   V_p,y = lambda x |model prediction for that shock on p's profile of year y|, in %.
4. Interval: 5th-95th percentiles over draws that combine a bootstrap refit of the model
   and a resample of the calibration departures (§10.4 step 5).
5. v_usd = V x the company's mean market cap of the year. Pi (§3.6, model-free check) is the
   share of the year's squared market-adjusted returns that falls on the person's shock
   windows (J0 to J0+2).

Run: python -m kpr.scores.value   (writes processed/scores/value.parquet)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kpr.build.features import load_companies
from kpr.models.baselines import load_features
from kpr.models.cv import load_ml_params, modelling_frame
from kpr.scores.sensitivity import (
    bootstrap_predict,
    load_predictive_model,
    shocked,
    window_profiles,
)
from kpr.utils.io import data_dir, read_table, use_fixtures
from kpr.utils.logging import get_logger

log = get_logger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
MATCH_DAYS = 10  # an event within 10 days of the news belongs to the departure (§11.2)
SEED = 42


def load_departures(path: Path = CONFIG_DIR / "natural_experiments.csv") -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["wanted_by_company"] = df["wanted_by_company"].str.lower().eq("true")
    return df


def calibration_events(departures: pd.DataFrame, events: pd.DataFrame,
                       calibration_end: pd.Timestamp) -> pd.DataFrame:
    """Events of the unwanted departures dated up to `calibration_end` (one per departure:
    the event of the same company within 10 days with the largest attention spike)."""
    keep = departures[~departures["wanted_by_company"] & (departures["date"] <= calibration_end)]
    rows = []
    for d in keep.itertuples(index=False):
        cand = events[(events["company_id"] == d.company_id)
                      & ((events["j0"] - d.date).abs() <= np.timedelta64(MATCH_DAYS, "D"))
                      & events["car_0_2"].notna()]
        if len(cand):
            rows.append(cand.sort_values("z_wiki", ascending=False).iloc[0])
    out = pd.DataFrame(rows).reset_index(drop=True)
    log.info("lambda calibration: %d unwanted departures up to %s, %d matched to an event",
             len(keep), calibration_end.date(), len(out))
    return out


def departure_value(model, profiles: pd.DataFrame, departures: pd.DataFrame,
                    features: pd.DataFrame, feat_cols: list[str],
                    X_train: pd.DataFrame | None = None, y_train: pd.Series | None = None,
                    n_model_boot: int = 0, n_draws: int = 0, default_z: float = 4.0,
                    seed: int = SEED) -> pd.DataFrame:
    """V in % per profile row, with lambda and (if n_model_boot > 0) a bootstrap interval."""
    out = profiles[["person_id", "company_id", "year"]].copy()
    if departures.empty:
        log.warning("no calibration departure: lambda set to 1 (V not calibrated)")
        lam, dep_X, actual = 1.0, None, None
        z, tone, spec = default_z, profiles["tone_q10"], None  # standard negative shock
    else:
        dep_X = departures[["event_id"]].merge(features, on="event_id")[feat_cols]
        actual = departures["car_0_2"].abs().to_numpy()
        lam = float(actual.mean() / max(np.abs(model.predict(dep_X)).mean(), 1e-12))
        z = float(departures["z_wiki"].mean())
        tone = float(departures["tone_rel"].mean())
        spec = float(dep_X["specificity"].mean()) if "specificity" in dep_X else None
    shock = shocked(profiles, feat_cols, z, tone, spec)
    out["v_pct"] = 100 * lam * np.abs(model.predict(shock))
    out["lambda"], out["lambda_n"] = lam, len(departures)
    log.info("lambda = %.2f on %d departures", lam, len(departures))

    if n_model_boot and X_train is not None:
        mats = {"shock": shock} if dep_X is None else {"shock": shock, "dep": dep_X}
        boot = bootstrap_predict(model, X_train, y_train, mats, n_model_boot, seed)
        rng = np.random.default_rng(seed + 1)
        draws = np.empty((n_draws, len(out)))
        for j in range(n_draws):
            b = j % n_model_boot
            if dep_X is None:
                lam_j = 1.0
            else:
                idx = rng.integers(0, len(actual), len(actual))
                lam_j = actual[idx].mean() / max(np.abs(boot["dep"][b][idx]).mean(), 1e-12)
            draws[j] = 100 * lam_j * np.abs(boot["shock"][b])
        out["v_lo"] = np.quantile(draws, 0.05, axis=0)
        out["v_hi"] = np.quantile(draws, 0.95, axis=0)
    return out


def mean_market_cap(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices.assign(year=prices["date"].dt.year,
                      mcap=prices["close_adj"] * prices["shares_out"])
    return p.groupby(["company_id", "year"], as_index=False)["mcap"].mean()


def pi_share(events: pd.DataFrame, prices: pd.DataFrame, market: pd.DataFrame,
             companies: pd.DataFrame) -> pd.DataFrame:
    """Pi of §3.6 per (person, company, year), on market-adjusted returns (beta = 1)."""
    index_of = companies.set_index("company_id")["market_index"].to_dict()
    rows = []
    for cid, ev in events.groupby("company_id"):
        if cid not in index_of:
            continue
        px = prices[prices["company_id"] == cid].set_index("date")["ret"]
        mk = market[market["index_id"] == index_of[cid]].set_index("date")["ret"]
        ar = (px - mk.reindex(px.index)).dropna().sort_index()
        for (pid, year), g in ev.groupby(["person_id", ev["j0"].dt.year]):
            year_ar = ar[ar.index.year == year]
            total = float((year_ar ** 2).sum())
            shock_days: set = set()
            for j0 in g["j0"]:
                pos = year_ar.index.searchsorted(j0)
                shock_days.update(year_ar.index[pos:pos + 3])
            on_shocks = float((year_ar.loc[sorted(shock_days)] ** 2).sum())
            rows.append({"person_id": pid, "company_id": cid, "year": int(year),
                         "pi_share": on_shocks / total if total > 0 else np.nan})
    return pd.DataFrame(rows, columns=["person_id", "company_id", "year", "pi_share"])


def build_values(events: pd.DataFrame, features: pd.DataFrame, params: dict,
                 splits: tuple[str, ...], tables: dict[str, pd.DataFrame],
                 departures: pd.DataFrame, companies: pd.DataFrame, model=None,
                 n_model_boot: int | None = None) -> pd.DataFrame:
    feat_cols = [c for c in features.columns if c != "event_id"]
    frame = modelling_frame(events, features, splits)
    train = modelling_frame(events, features, ("train",)).dropna(subset=["car_abs"])
    model = model or load_predictive_model(feat_cols, train, "car_signed")
    years = sorted(frame["j0"].dt.year.unique().tolist())
    profiles = window_profiles(frame, feat_cols, years, params["scores"]["window_years"])
    quant = yaml.safe_load((CONFIG_DIR / "params" / "quant.yaml").read_text(encoding="utf-8"))
    calib = calibration_events(departures, events, pd.Timestamp(quant["splits"]["train"][1]))
    sc = params["scores"]
    n_model_boot = sc["bootstrap_samples"] if n_model_boot is None else n_model_boot
    out = departure_value(model, profiles, calib, features, feat_cols, train[feat_cols],
                          train["car_signed"], n_model_boot,
                          sc["value_bootstrap_samples"] if n_model_boot else 0,
                          params["standard_shock"]["z_wiki"])
    out = out.merge(mean_market_cap(tables["prices"]), on=["company_id", "year"], how="left")
    out["v_usd"] = out["v_pct"] / 100 * out.pop("mcap")
    pi = pi_share(frame, tables["prices"], tables["market"], companies)
    return out.merge(pi, on=["person_id", "company_id", "year"], how="left")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lambda calibration and V (§10.4)")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--n-boot", type=int, default=None)
    args = parser.parse_args(argv)
    params = load_ml_params()
    events = read_table("events")
    tables = {n: read_table(n) for n in ("prices", "market")}
    out = build_values(events, load_features(events), params, tuple(args.splits), tables,
                       load_departures(), load_companies(), n_model_boot=args.n_boot)
    log.info("V: median %.2f%%, lambda %.2f", out["v_pct"].median(), out["lambda"].iloc[0])
    if use_fixtures():
        log.info("fixture run: values not written")
        return 0
    path = data_dir() / "processed" / "scores" / "value.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    log.info("written to %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
