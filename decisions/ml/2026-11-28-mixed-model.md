Date: 2026-11-28
Session: ml
Decision: the hierarchical model is a statsmodels MixedLM fitted by REML with person groups,
random slopes on tone_neg and tone_pos and no random intercept (exactly the §9.4 formula),
fixed effects for the tone parts and for controls [z_wiki, z_news, vix, vol_20d, log_mcap,
mkt_ret_j0, car_m5_m1] (ml.yaml `mixed.controls`), each median-imputed and standardised on
the estimation sample; the tone parts stay in raw units so that slopes read as CAR per unit
of relative tone. Persons with fewer than 3 events in a window are left out of that fit.
One fit per 3-year window [y-2, y] gives slopes per person and year (§10.2, §3.4). Optimiser:
L-BFGS then Powell; convergence warnings are logged and a `converged` flag is stored.
By default the module estimates on the training split; which splits feed the published
yearly scores is decided in M10 (the test split only after the M12 go).
Why: dossier §9.4 (CAR = gamma'x + (beta- + b-_p) tone- + (beta+ + b+_p) tone+ + eps) and
§10.2 (method B uses beta + b_p; re-estimated on each rolling 3-year window).
Alternatives considered: adding a random intercept (configurable via
`mixed.random_intercept`, off to match the dossier and keep the slopes the only
person-specific part); Bayesian hierarchical model (heavier, no gain for method B);
standardising the tone parts (rejected: the standard shock of §3.1 is expressed in tone units).
