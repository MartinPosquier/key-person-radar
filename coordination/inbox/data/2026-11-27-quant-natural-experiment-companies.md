From: quant
To: data
Need: (1) use the same `company_id` values as `config/natural_experiments.csv`, which uses the yfinance ticker (JPM, TTE.PA, STLAM.MI, BRK-B, NOVO-B.CO...), or tell me the mapping once universe.yaml is final, and I will align the CSV. (2) Prices and Wikipedia pageviews (2014-2025) for three companies outside the universe that host natural experiments: MCD (Steve Easterbrook, 2019-11-03), CMG (Brian Niccol, 2024-08-13), UNH (Brian Thompson, 2024-12-04). They are not added to the 60-company study; they only serve the ex ante validation (dossier §11.2: "le pipeline est générique").
Also: check that yfinance has the FCA history under STLAM.MI before 2021 (event NE004, 2018-07-21); otherwise FCA.MI or the Milan ticker of the time.
Why: dossier §11.2 (at least ~20 natural experiments, extended beyond the 60 companies). Task Q3 lists 28 candidates.
Urgency: low (needed for Q10 in week S4; Q5 only needs the dates to flag reserved events)
