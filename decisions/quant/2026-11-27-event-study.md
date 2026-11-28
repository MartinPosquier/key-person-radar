Date: 2026-11-27
Session: quant
Decision: the event study of §7.6 uses kept events of the train and validation splits only; tests: one-sample Student t and two-sided sign test (binomial) on CAR[0, 2] by sign; media vs control compared on |CAR[0, 2]| with one-sided Welch and Mann-Whitney tests; CAR curve J-10..J+10 with a 95 % band (mean ± 1.96 standard errors)
Why: §7.6 lists exactly these four outputs. The test period (2024-2025) is "utilisée une seule fois" (§9.5), so even descriptive statistics leave it alone until task M12; natural experiments stay reserved for §11.2. One-sided tests for media vs control because the dossier's question is directional ("l'ampleur moyenne |CAR| est-elle plus forte dans le premier groupe ?").
Code layout: logic in kpr.validation.event_study (tested), notebook 02_event_study only calls it and is committed without outputs.
Alternatives considered: Patell or BMP standardised tests (possible later as robustness; the dossier asks for t and sign tests).
