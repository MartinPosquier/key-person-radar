Date: 2026-11-28
Session: ml
Decision: the four model families of §9.4 are Lasso, Elastic Net, Random Forest and
scikit-learn's HistGradientBoosting (regression), with logistic regression (elastic-net
path via l1_ratio), Random Forest (balanced class weights) and HistGradientBoosting for the
big-move task. Hyperparameters are tuned by grid search on the training split only, with
the expanding yearly folds of kpr.models.cv, scored by MAE (regression) or average
precision (classification); grids live in config/params/ml.yaml (`models.grids`). The tuned
models are refitted on the whole training split and compared with B0-B2 on the validation
split, which chooses the final model per target (lowest validation MAE, highest PR-AUC).
Linear models get median imputation + standardisation, forests median imputation, and
HistGradientBoosting reads missing values natively (only columns empty in the whole fold
are dropped). Every step is inside the pipeline, so it is fitted on the training fold only.
Why: dossier §9.4 (models and their main settings), §9.5 (tuning by internal CV on
2015-2021, model choice on 2022-2023, test untouched), §9.2 (scaling inside the pipeline).
Alternatives considered: LightGBM (kept optional by §9.4; HistGradientBoosting gives the
same family without a second API and handles NaN the same way); randomised search
(unnecessary at this grid size); nested CV for an unbiased CV score (the validation split
already plays that role).
