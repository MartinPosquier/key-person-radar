"""Export result files for the Streamlit app into app/data/.

Owner: ml session. Task M13 in PARALLEL_SESSIONS.md §7. Dossier §12.3, §12.4, §10.7.

The app never computes anything (§12.3): this command copies the precomputed results it
needs into app/data/, in the format checked by app/common.py:
  scores.parquet    the `scores` table
  events.parquet    the `events` table plus `car_pred`, the best model's prediction
  daily.parquet     views, adjusted close and market-adjusted return of tracked persons
  entities.parquet  display names, role, founder, region, sector, group
  meta.json         origin, export date, and the sanity checks of §10.7
  case_study.json   the Nvidia 2026 case (quant task Q12), when it exists

Run: python -m kpr.scores.export_app [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kpr.utils.logging import get_logger

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_DATA = REPO_ROOT / "app" / "data"
LOW_CONFIDENCE = 8  # dossier §10.7, §12.4


def _yaml_list(name: str, key: str) -> list[dict]:
    raw = yaml.safe_load((REPO_ROOT / "config" / name).read_text(encoding="utf-8")) or {}
    return raw.get(key) or []


def build_entities(scores: pd.DataFrame, mandates: pd.DataFrame, persons: list[dict],
                   companies: list[dict]) -> pd.DataFrame:
    """One row per tracked (person, company): names from the configs, role from mandates."""
    pairs = scores[["person_id", "company_id"]].drop_duplicates()
    names = {}
    for p in persons:  # the English Wikipedia title is the display name
        title = (p.get("wiki_pages") or {}).get("en")
        names[p["person_id"]] = p.get("name") or (title.replace("_", " ") if title else None)
    comp = {c["company_id"]: c for c in companies}
    last = mandates.sort_values("start").drop_duplicates(["person_id", "company_id"], keep="last")
    out = pairs.merge(last[["person_id", "company_id", "role", "founder"]], how="left")
    out["person_name"] = out["person_id"].map(names).fillna(
        out["person_id"].str.replace("_", " ").str.title())
    for col, key, default in (("company_name", "name", None), ("region", "region", "unknown"),
                              ("sector", "sector", "unknown"), ("group", "group", "unknown")):
        out[col] = out["company_id"].map(lambda c, k=key: comp.get(c, {}).get(k))
        out[col] = out[col].fillna(out["company_id"] if default is None else default)
    out["role"] = out["role"].fillna("other")
    out["founder"] = out["founder"].fillna(False).astype(bool)
    return out[["person_id", "person_name", "company_id", "company_name", "role", "founder",
                "region", "sector", "group"]]


def build_daily(panel: pd.DataFrame, prices: pd.DataFrame, market: pd.DataFrame,
                companies: list[dict], persons: pd.Series) -> pd.DataFrame:
    """Daily series of tracked persons; ar = stock return - its index return (beta = 1)."""
    index_of = {c["company_id"]: c.get("market_index") for c in companies}
    px = prices[["company_id", "date", "close_adj", "ret"]].copy()
    px["index_id"] = px["company_id"].map(index_of)
    mk = market[["index_id", "date", "ret"]].rename(columns={"ret": "ret_mkt"})
    px = px.merge(mk, on=["index_id", "date"], how="left")
    px["ar"] = px["ret"] - px["ret_mkt"]
    daily = panel.loc[panel["person_id"].isin(persons), ["person_id", "date", "company_id",
                                                         "views"]]
    daily = daily.merge(px[["company_id", "date", "close_adj", "ar"]], how="left")
    return daily.sort_values(["person_id", "date"]).reset_index(drop=True)


def build_events(events: pd.DataFrame, car_pred: pd.Series | None) -> pd.DataFrame:
    out = events.copy()
    out["car_pred"] = out["event_id"].map(car_pred) if car_pred is not None else np.nan
    return out


def case_from_quant(case: dict) -> dict:
    """Convert processed/validation/case_nvidia.json (Q12) into the app's case_study.json."""
    model = case.get("model", {})
    sign = -1.0 if model.get("sign", "negative") == "negative" else 1.0
    actual = case.get("cars", {}).get(case["company"], {}).get("car_0_2")
    news = case.get("news", {})
    story = (f"On {case['date']}, attention on {case['person'].replace('_', ' ').title()} "
             f"peaked ({case.get('date_source', '')}). The model's end-of-"
             f"{model.get('year')} scores implied a reaction of about "
             f"{100 * model.get('expected_abs_car', float('nan')):.1f}% in absolute value. "
             f"Verdict: {case.get('verdict')}. Earnings dates nearby: "
             f"{', '.join(news.get('earnings_dates', [])) or 'none'}.")
    return {"title": f"{case['company']}, {case['date']}", "person_id": case["person"],
            "company_id": case["company"], "event_date": case["j0"], "story": story,
            "predicted_car": sign * model.get("expected_abs_car", float("nan")),
            "actual_car": actual, "demo": False}


def sanity_checks(scores: pd.DataFrame, entities: pd.DataFrame) -> dict:
    """§10.7 checks, recorded in meta.json and logged (never silently 'fixed')."""
    latest = scores.loc[scores.groupby(["person_id", "company_id"])["year"].idxmax()]
    latest = latest.merge(entities[["person_id", "company_id", "group"]], how="left")
    top = latest.sort_values("v_pct", ascending=False).head(10)
    by_group = latest.groupby("group")["v_pct"].mean().round(3).to_dict()
    suspicious = latest[(latest["v_pct"] > latest["v_pct"].quantile(0.75))
                        & (latest["pi_share"] < 0.01)]
    checks = {
        "top10_control_group": int((top["group"] == "control").sum()),
        "mean_v_by_group": by_group,
        "low_confidence_person_years": int((scores["n_events"] < LOW_CONFIDENCE).sum()),
        "high_v_low_pi": suspicious["person_id"].tolist(),
    }
    if checks["top10_control_group"]:
        log.warning("control-group persons in the top 10 by V: look for a bug first (§10.7)")
    if checks["high_v_low_pi"]:
        log.warning("high V with Pi ~ 0, to investigate: %s", checks["high_v_low_pi"])
    return checks


def export(out_dir: Path, tables: dict[str, pd.DataFrame], persons: list[dict],
           companies: list[dict], car_pred: pd.Series | None, case: dict | None,
           source: str, demo: bool = False) -> dict:
    scores = tables["scores"]
    entities = build_entities(scores, tables["mandates"], persons, companies)
    frames = {
        "scores": scores,
        "events": build_events(tables["events"], car_pred),
        "daily": build_daily(tables["panel"], tables["prices"], tables["market"], companies,
                             scores["person_id"].unique()),
        "entities": entities,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    meta = {"source": source, "demo": demo, "exported_at": date.today().isoformat(),
            "years": [int(scores["year"].min()), int(scores["year"].max())],
            "checks": sanity_checks(scores, entities)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    case_path = out_dir / "case_study.json"
    if case is not None:
        case_path.write_text(json.dumps(case_from_quant(case), indent=2), encoding="utf-8")
    elif case_path.exists():
        case_path.unlink()  # never leave the demo case next to real results
    sizes = {n: f"{(out_dir / f'{n}.parquet').stat().st_size / 1e6:.2f} MB" for n in frames}
    log.info("exported to %s: %s", out_dir, sizes)
    return meta


def best_model_predictions(events: pd.DataFrame) -> pd.Series | None:
    """car_pred per event from the M7 best model, if the models exist."""
    import joblib

    from kpr.models.baselines import load_features
    from kpr.utils.io import data_dir

    models = data_dir() / "processed" / "models"
    best = models / "best_models.json"
    if not best.exists():
        log.warning("no M7 model: car_pred left empty")
        return None
    name = json.loads(best.read_text(encoding="utf-8"))["car_signed"]
    model = joblib.load(models / f"car_signed__{name}.joblib")
    features = load_features(events)
    cols = [c for c in features.columns if c != "event_id"]
    return pd.Series(model.predict(features[cols]), index=features["event_id"])


def main(argv: list[str] | None = None) -> int:
    from kpr.utils.io import data_dir, read_table, use_fixtures

    parser = argparse.ArgumentParser(description="Export app data (dossier §12.3)")
    parser.add_argument("--out", type=Path, default=None, help="default: app/data/")
    args = parser.parse_args(argv)
    if use_fixtures() and args.out is None:
        log.error("fixture run: pass --out so the app's data folder is not overwritten")
        return 2
    tables = {n: read_table(n) for n in ("scores", "events", "mandates", "panel", "prices",
                                         "market")}
    case_path = data_dir() / "processed" / "validation" / "case_nvidia.json"
    case = (json.loads(case_path.read_text(encoding="utf-8"))
            if case_path.exists() and not use_fixtures() else None)
    car_pred = None if use_fixtures() else best_model_predictions(tables["events"])
    export(args.out or APP_DATA, tables, _yaml_list("persons.yaml", "persons"),
           _yaml_list("universe.yaml", "companies"), car_pred, case,
           source="synthetic test fixtures" if use_fixtures() else "pipeline results",
           demo=use_fixtures())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
