Date: 2026-11-27
Session: ml
Decision: (1) CV folds are calendar years with an expanding window: train up to year y-1,
validate on year y, for y from cv.first_train_end_year + 1 (2018) to the last training
year (2021), i.e. 4 folds. The 10-session gap drops training events whose J0 is within 10
business days of January 1 of the validation year. (2) B0 for the absolute target is the
mean |CAR[0,2]| of the training events of the same company (overall training mean for an
unseen company). (3) The big-move label is |CAR[0,2]| > 2 x vol_20d x sqrt(3), i.e. two
standard deviations of the usual 3-session move. (4) The CV harness refuses the test
split unless `allow_test=True` (task M12 only); natural experiments and excluded events
never enter training.
Why: dossier §9.5 describes yearly expanding folds (train 2015-2017 / validate 2018, ...)
with a 10-session gap; §9.3 defines B0 for the absolute target as the stock's historical
mean absolute 3-day abnormal return, but no frozen table holds daily abnormal returns, so
the training events of the company are the closest leak-free proxy; §9.1 defines the
classification target relative to the usual volatility.
Alternatives considered: sklearn TimeSeriesSplit(gap=...) on rows sorted by date (rejected:
its folds have equal row counts, not calendar years, so the gap in rows is not 10
sessions); counting the gap with the quant trading calendar (switch to it once Q1 is
merged; business days differ only on exchange holidays).
