Date: 2026-11-28
Session: ml
Decision: the four §9.6 methods are run as follows. Filter: mutual information
(SelectKBest, k = 10) and a greedy mRMR whose redundancy term is the mean absolute Pearson
correlation with the features already chosen (MID criterion). Wrapper: RFE around a Random
Forest (200 trees, min 20 events per leaf), k = 10. Embedded: the Lasso path on
standardised training data, summarised per feature as the largest alpha at which it is
still non-zero. Stability: Lasso (alpha = the tuned Lasso alpha from M7) refitted on 100
bootstrap resamples, scaling refitted inside each resample; a feature kept in more than
80 % of fits is "stable". Every selector sits inside a pipeline (impute -> scale -> select
-> model) evaluated with the expanding yearly folds, so selection never sees the validation
year; the report also gives, per method, how often each feature was kept across folds.
Why: dossier §9.6 (filter, wrapper, embedded, stability; 100 bootstraps; 80 % threshold;
"selection inside the CV pipeline, never before", LAB 4 Q5 and Q7).
Alternatives considered: mRMR with mutual information as redundancy (quadratic number of MI
estimates, slow and noisy on a few thousand rows); RFECV choosing k by CV (kept simple with
a fixed k from ml.yaml; the fold frequencies already show which features matter).
