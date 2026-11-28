"""Build the demo files of app/data/ from the synthetic test fixtures (task M1).

The app reads precomputed files only (dossier §12.3). Until real scores exist (task M10),
this script fills app/data/ with fixture-shaped data so that every page can be built,
tested and deployed. `meta.json` carries `"demo": true`, which makes every page show a
"demo mode" banner. The real export (task M13) overwrites these files.

Run: python scripts/ml_make_app_demo.py
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["KPR_USE_FIXTURES"] = "1"

from kpr.utils.io import read_table  # noqa: E402  (must follow the fixture switch)

APP_DATA = Path(__file__).resolve().parents[1] / "app" / "data"

# Display metadata for the fake fixture entities. Real names come from the YAML configs.
DEMO_PERSONS = {
    "p_alpha": "Alex Alpha",
    "p_beta": "Bea Beta",
    "p_gamma": "Gil Gamma",
    "p_delta": "Dora Delta",
    "p_eps": "Eli Epsilon",
}
DEMO_COMPANIES = {  # company_id -> (name, region, sector, market index, group)
    "FAKE1": ("Fakecorp Semis", "US", "Semiconductors", "^GSPC", "media"),
    "FAKE2": ("Demo Motors", "US", "Automobiles", "^GSPC", "media"),
    "FAKE3": ("Synthetic Energies", "EU", "Energy", "^STOXX", "control"),
}


def build_entities(mandates: pd.DataFrame) -> pd.DataFrame:
    ent = mandates[["person_id", "company_id", "role", "founder"]].copy()
    ent["person_name"] = ent["person_id"].map(DEMO_PERSONS)
    for i, col in enumerate(["company_name", "region", "sector"]):
        ent[col] = ent["company_id"].map({k: v[i] for k, v in DEMO_COMPANIES.items()})
    ent["group"] = ent["company_id"].map({k: v[4] for k, v in DEMO_COMPANIES.items()})
    return ent[["person_id", "person_name", "company_id", "company_name", "role", "founder",
                "region", "sector", "group"]]


def build_daily(panel: pd.DataFrame, prices: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Daily views, adjusted close and market-adjusted abnormal return per tracked person."""
    index_of = {k: v[3] for k, v in DEMO_COMPANIES.items()}
    px = prices[["company_id", "date", "close_adj", "ret"]].copy()
    px["index_id"] = px["company_id"].map(index_of)
    mkt = market[["index_id", "date", "ret"]].rename(columns={"ret": "ret_mkt"})
    px = px.merge(mkt, on=["index_id", "date"], how="left")
    px["ar"] = px["ret"] - px["ret_mkt"]
    daily = panel[["person_id", "date", "company_id", "views"]].merge(
        px[["company_id", "date", "close_adj", "ar"]], on=["company_id", "date"], how="left"
    )
    return daily.sort_values(["person_id", "date"]).reset_index(drop=True)


def build_events(events: pd.DataFrame) -> pd.DataFrame:
    """Shared events plus a demo `car_pred`: a linear fit of CAR on tone, train split only."""
    out = events.copy()
    train = out[(out["split"] == "train") & ~out["excluded"]]
    slope, intercept = np.polyfit(train["tone_rel"], train["car_0_2"], 1)
    out["car_pred"] = intercept + slope * out["tone_rel"]
    return out


def build_case_study(events: pd.DataFrame, entities: pd.DataFrame) -> dict:
    """A placeholder case on the largest negative fixture shock, flagged as demo."""
    neg = events[events["sign"] == "negative"].sort_values("car_0_2").iloc[0]
    name = entities.set_index("person_id").loc[neg.person_id, "person_name"]
    return {
        "title": f"Demo: {name}, {neg.j0:%d %B %Y}",
        "person_id": neg.person_id,
        "company_id": neg.company_id,
        "event_date": f"{neg.j0:%Y-%m-%d}",
        "story": "Placeholder story on synthetic data. The real Nvidia 2026 case "
        "(dossier §11.6) replaces it with the final results.",
        "predicted_car": float(neg.car_pred),
        "actual_car": float(neg.car_0_2),
        "demo": True,
    }


def main() -> None:
    mandates = read_table("mandates")
    entities = build_entities(mandates)
    frames = {
        "scores": read_table("scores"),
        "events": build_events(read_table("events")),
        "daily": build_daily(read_table("panel"), read_table("prices"), read_table("market")),
        "entities": entities,
    }
    APP_DATA.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_parquet(APP_DATA / f"{name}.parquet", index=False)
        print(f"{name:<9} {len(df):>6} rows")
    case = build_case_study(frames["events"], entities)
    (APP_DATA / "case_study.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
    meta = {"source": "synthetic test fixtures", "demo": True,
            "exported_at": date.today().isoformat()}
    (APP_DATA / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"written to {APP_DATA}")


if __name__ == "__main__":
    main()
