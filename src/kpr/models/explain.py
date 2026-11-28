"""SHAP values and analysis of the 20 largest errors.

Owner: ml session. Task M11 in PARALLEL_SESSIONS.md §7. Dossier §9.8, §9.9.

Interpretability (§9.8):
- SHAP values of the best tree model (TreeExplainer on the fitted pipeline's estimator),
  summarised as mean |SHAP| per feature and per family, plus a summary plot and a
  dependence plot (how the predicted CAR moves with tone, coloured by attention share);
- Lasso coefficients, in points of CAR per standard deviation of each feature;
- a local explanation of any event (`--event <event_id>`), e.g. Ghosn 2018.
Error analysis (§9.9), on the validation split only (the test split waits for M12):
- MAE by sector, region, role, sign and year;
- the 20 largest absolute errors, one row each, with what is needed to review them by hand.

Run: python -m kpr.models.explain [--event ID]   (writes processed/models/explain/)
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from kpr.build.features import FAMILIES, family_columns
from kpr.models.baselines import load_features
from kpr.models.cv import load_ml_params, modelling_frame
from kpr.models.tune import make_model
from kpr.utils.io import data_dir, read_table, use_fixtures
from kpr.utils.logging import get_logger

log = get_logger(__name__)

N_LARGEST = 20
GROUPS = ("sector", "region", "role")


# ---- SHAP -----------------------------------------------------------------------------------


def transformed(pipe, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Matrix seen by the pipeline's estimator, and its column names."""
    prep = pipe.named_steps["prep"]
    Z = prep.transform(X)
    cols = list(prep.transformers_[0][2])
    step = prep.named_transformers_["num"]
    keep = getattr(step, "keep_", None)  # DropEmptyColumns of the HGB pipeline
    names = [c for c, k in zip(cols, keep, strict=True) if k] if keep is not None else cols
    return np.asarray(Z, float), names


def shap_values(pipe, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP values (one column per feature) of a fitted tree pipeline on X."""
    import shap

    Z, names = transformed(pipe, X)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    values = explainer.shap_values(Z)
    out = pd.DataFrame(values, columns=names, index=X.index)
    out.attrs["expected_value"] = float(np.ravel(explainer.expected_value)[0])
    return out


def shap_summary(values: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per feature with its family, sorted."""
    imp = values.abs().mean().rename("mean_abs_shap").to_frame()
    imp["family"] = [next((f for f in FAMILIES if c in family_columns([c], f)), "other")
                     for c in imp.index]
    imp["share"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    return imp.sort_values("mean_abs_shap", ascending=False)


def local_explanation(values: pd.DataFrame, X: pd.DataFrame, row) -> pd.DataFrame:
    """Contribution of each feature to one prediction, largest first."""
    contrib = values.loc[row]
    out = pd.DataFrame({"value": X.loc[row, contrib.index], "shap": contrib})
    out.attrs["expected_value"] = values.attrs.get("expected_value")
    return out.reindex(out["shap"].abs().sort_values(ascending=False).index)


def lasso_coefficients(pipe) -> pd.Series:
    """Lasso weights on standardised features: CAR points (x100) per standard deviation."""
    names = list(pipe.named_steps["prep"].transformers_[0][2])
    coef = pd.Series(100 * pipe.named_steps["model"].coef_, index=names, name="car_pts_per_sd")
    return coef[coef != 0].sort_values(key=np.abs, ascending=False)


def save_plots(values: pd.DataFrame, X: pd.DataFrame, out_dir) -> None:
    """Summary plot and tone dependence plot (coloured by attention share) as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    feats = X[values.columns]
    shap.summary_plot(values.to_numpy(), feats, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=150)
    plt.close("all")
    if "tone_rel" in values.columns:
        colour = "attention_share" if "attention_share" in values.columns else None
        shap.dependence_plot("tone_rel", values.to_numpy(), feats, interaction_index=colour,
                             show=False)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_dependence_tone.png", dpi=150)
        plt.close("all")


# ---- error analysis -------------------------------------------------------------------------


def _label(frame: pd.DataFrame, prefix: str) -> pd.Series:
    """Category of a one-hot family (e.g. role_CEO -> CEO), 'unknown' when absent."""
    cols = [c for c in frame.columns if c.startswith(prefix + "_")]
    if not cols:
        return pd.Series("unknown", index=frame.index)
    return frame[cols].idxmax(axis=1).str.removeprefix(prefix + "_")


def error_analysis(frame: pd.DataFrame, pred: np.ndarray, target: str = "car_signed"):
    """MAE by group and year, and the N_LARGEST largest absolute errors (§9.9)."""
    df = frame.assign(pred=pred, error=pred - frame[target], abs_error=np.abs(pred - frame[target]),
                      year=frame["j0"].dt.year)
    for g in GROUPS:
        df[g] = _label(df, g)
    by_group = pd.concat(
        [df.groupby(g)["abs_error"].agg(mae="mean", n="size").reset_index()
         .rename(columns={g: "group"}).assign(dimension=g)
         for g in (*GROUPS, "sign", "year")], ignore_index=True,
    )[["dimension", "group", "n", "mae"]]
    cols = ["event_id", "person_id", "company_id", "j0", "sign", "tone_rel", "z_wiki",
            target, "pred", "error", *GROUPS]
    largest = df.nlargest(N_LARGEST, "abs_error")[[c for c in cols if c in df.columns]]
    largest["review_note"] = ""  # filled by hand: competing news? missed earnings? ...
    return by_group, largest.reset_index(drop=True)


# ---- pipeline -------------------------------------------------------------------------------


def explain(events: pd.DataFrame, features: pd.DataFrame, params: dict, pipes: dict,
            event_id: str | None = None) -> dict:
    """All §9.8-9.9 outputs as DataFrames. `pipes` = {"tree": ..., "lasso": ...} fitted."""
    feat_cols = [c for c in features.columns if c != "event_id"]
    valid = modelling_frame(events, features, ("validation",))
    out: dict = {}
    values = shap_values(pipes["tree"], valid[feat_cols])
    out["shap_values"], out["shap_summary"] = values, shap_summary(values)
    out["family_importance"] = out["shap_summary"].groupby("family")["share"].sum() \
        .sort_values(ascending=False)
    if "lasso" in pipes:
        out["lasso_coefficients"] = lasso_coefficients(pipes["lasso"])
    if event_id is not None:
        row = valid.index[valid["event_id"] == event_id]
        if len(row):
            out["local"] = local_explanation(values, valid[feat_cols], row[0])
        else:
            log.warning("event %s is not a validation event: no local explanation", event_id)
    out["errors_by_group"], out["largest_errors"] = error_analysis(
        valid, pipes["tree"].predict(valid[feat_cols]))
    return out


def load_pipes(feat_cols: list[str], train: pd.DataFrame, target: str = "car_signed") -> dict:
    """Tree and Lasso pipelines of M7 (best tree model by validation MAE), or quick fits."""
    import joblib

    models = None if use_fixtures() else data_dir() / "processed" / "models"
    comparison = models / "model_comparison.csv" if models is not None else None
    if comparison is not None and comparison.exists():
        comp = pd.read_csv(comparison).query("target == @target and model in ['rf', 'hgb']")
        tree = comp.sort_values("mae")["model"].iloc[0]
        log.info("SHAP on the best tree model of M7: %s", tree)
        return {"tree": joblib.load(models / f"{target}__{tree}.joblib"),
                "lasso": joblib.load(models / f"{target}__lasso.joblib")}
    log.warning("no M7 models: quick untuned fits (fixtures only)")
    X, y = train[feat_cols], train[target]
    return {"tree": make_model("hgb", feat_cols).fit(X, y),
            "lasso": make_model("lasso", feat_cols).set_params(model__alpha=1e-4).fit(X, y)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SHAP and error analysis (§9.8, §9.9)")
    parser.add_argument("--event", default=None, help="event_id to explain locally")
    args = parser.parse_args(argv)
    params = load_ml_params()
    events = read_table("events")
    features = load_features(events)
    feat_cols = [c for c in features.columns if c != "event_id"]
    train = modelling_frame(events, features, ("train",)).dropna(subset=["car_signed"])
    out = explain(events, features, params, load_pipes(feat_cols, train), args.event)
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        log.info("importance by family:\n%s", out["family_importance"].round(3).to_string())
        log.info("top features:\n%s", out["shap_summary"].head(10).round(5).to_string())
        log.info("largest errors:\n%s", out["largest_errors"].head(5).to_string(index=False))
    if use_fixtures():
        log.info("fixture run: explanations not written")
        return 0
    out_dir = data_dir() / "processed" / "models" / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)
    out["shap_summary"].to_csv(out_dir / "shap_summary.csv", index_label="feature")
    out["family_importance"].to_csv(out_dir / "family_importance.csv")
    out["errors_by_group"].to_csv(out_dir / "errors_by_group.csv", index=False)
    out["largest_errors"].to_csv(out_dir / "largest_errors.csv", index=False)
    if "lasso_coefficients" in out:
        out["lasso_coefficients"].to_csv(out_dir / "lasso_coefficients.csv",
                                         index_label="feature")
    if "local" in out:
        out["local"].to_csv(out_dir / f"local_{args.event}.csv", index_label="feature")
    save_plots(out["shap_values"], modelling_frame(events, features, ("validation",)),
               out_dir)
    (out_dir / "meta.json").write_text(json.dumps(
        {"expected_value": out["shap_values"].attrs["expected_value"]}), encoding="utf-8")
    log.info("written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
