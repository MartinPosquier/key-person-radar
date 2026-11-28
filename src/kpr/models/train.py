"""Train Lasso / Elastic Net, Random Forest, gradient boosting.

Owner: ml session. Task M7 in PARALLEL_SESSIONS.md §7. Dossier §9.4.

Each model is refitted with its tuned hyperparameters (processed/models/tuning.json, from
kpr.models.tune) on the whole training split, then scored on the validation split next to
the baselines B0-B2. The validation split chooses the final model per target (§9.5); the
test split is never read here.

Outputs in processed/models/: one fitted pipeline per (target, model) as joblib,
validation_predictions.parquet, model_comparison.csv, best_models.json.

Run: python -m kpr.models.train
"""

from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from kpr.models.baselines import load_features, make_baselines, summarise
from kpr.models.cv import fit_predict, load_ml_params, modelling_frame
from kpr.models.tune import make_model, model_names, tune_all, tuning_path
from kpr.utils.io import data_dir, read_table, use_fixtures
from kpr.utils.logging import get_logger

log = get_logger(__name__)

TARGETS = ("car_signed", "car_abs", "big_move")


def choose_best(comparison: pd.DataFrame, regression_metric: str = "mae") -> dict[str, str]:
    """Best non-baseline model per target on validation: lowest error, or highest PR-AUC."""
    best = {}
    for target, rows in comparison[~comparison["baseline"]].groupby("target"):
        metric = "pr_auc" if target == "big_move" else regression_metric
        rows = rows.dropna(subset=[metric])
        if rows.empty:
            continue
        pick = rows[metric].idxmax() if metric in ("pr_auc", "r2") else rows[metric].idxmin()
        best[target] = rows.loc[pick, "model"]
    return best


def fit_and_compare(events: pd.DataFrame, features: pd.DataFrame, params: dict,
                    tuning: dict, targets: tuple[str, ...] = TARGETS):
    """Refit tuned models on train, score them and the baselines on validation."""
    feat_cols = [c for c in features.columns if c != "event_id"]
    train = modelling_frame(events, features, ("train",))
    valid = modelling_frame(events, features, ("validation",))
    rows, preds, fitted = [], [], {}
    for target in targets:
        task = "classification" if target == "big_move" else "regression"
        tr, va = train.dropna(subset=[target]), valid.dropna(subset=[target])
        if tr.empty or va.empty or target not in tuning:
            log.warning("%s: nothing to fit or to validate, skipped", target)
            continue
        y_tr = tr[target].astype(int) if task == "classification" else tr[target]
        candidates = {name: (model, True) for name, model in
                      make_baselines(feat_cols, target).items()}
        for name in model_names(task):
            model = make_model(name, feat_cols, task)
            model.set_params(**tuning[target][name]["best_params"])
            candidates[name] = (model, False)
        X_cols = [*feat_cols, "company_id"]  # B0 for |CAR| groups by company
        for name, (model, is_baseline) in candidates.items():
            proba = task == "classification"
            pred = fit_predict(model, tr[X_cols], y_tr, va[X_cols], proba)
            if not is_baseline:  # saved models see the feature columns only
                fitted[(target, name)] = model.fit(tr[feat_cols], y_tr)
            val = pd.DataFrame({"event_id": va["event_id"].to_numpy(), "target": target,
                                "model": name, "y_true": va[target].to_numpy(), "y_pred": pred})
            preds.append(val)
            rows.append({"target": target, "model": name, "baseline": is_baseline,
                         "n_train": len(tr), **summarise(val, target)})
    comparison = pd.DataFrame(rows)
    predictions = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return comparison, predictions, fitted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit tuned models, compare on validation")
    parser.parse_args(argv)
    params = load_ml_params()
    events = read_table("events")
    features = load_features(events)

    path = tuning_path() if not use_fixtures() else None
    if path is not None and path.exists():
        tuning = json.loads(path.read_text(encoding="utf-8"))
    else:
        log.info("no tuning.json: tuning now")
        tuning = tune_all(events, features, params)

    comparison, predictions, fitted = fit_and_compare(events, features, params, tuning)
    best = choose_best(comparison, params["models"]["selection_metric"])
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        log.info("validation comparison:\n%s", comparison.round(5).to_string(index=False))
    log.info("best model per target: %s", best)
    if use_fixtures():
        log.info("fixture run: models not written")
        return 0

    out = data_dir() / "processed" / "models"
    out.mkdir(parents=True, exist_ok=True)
    for (target, name), model in fitted.items():
        joblib.dump(model, out / f"{target}__{name}.joblib")
    predictions.to_parquet(out / "validation_predictions.parquet", index=False)
    comparison.to_csv(out / "model_comparison.csv", index=False)
    (out / "best_models.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    log.info("models, predictions and comparison written to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
