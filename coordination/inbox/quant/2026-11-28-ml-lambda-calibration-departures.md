From: ml
To: quant
Need: in Q10 (natural experiment validation), validate V only on natural experiments dated
after 2021-12-31, or report the pre-2022 ones separately as "used for calibration".
Why: dossier §10.4 calibrates lambda (the scaling factor of V) on the unwanted abrupt
departures of the training period, and says to split the list in two by date. M10
(`kpr.scores.value.calibration_events`) uses rows of config/natural_experiments.csv with
wanted_by_company = false and date <= the end of the training split (quant.yaml
splits.train[1]), matched to the same company's event within 10 days. Those events remain
reserved (never in model training); only lambda sees their CAR. S- and S+ do not use them,
so validating S- and S+ on all years stays clean.
Urgency: medium (needed before Q10 runs on real scores)
