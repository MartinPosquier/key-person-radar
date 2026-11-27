From: quant
To: user (contract), data (collection)
Need: a new shared table `ff_factors` (producer: data, folder: interim), keys **region, date**, columns region (string: "US" | "EU"), date, mkt_rf, smb, hml, rf (floats, decimals, not percent)
Source: Kenneth French data library, daily 3-factor files: "Fama/French 3 Factors [Daily]" for US and "Fama/French Developed Europe 3 Factors [Daily]" for EU (free CSV/zip download, no key). The parser already exists: `kpr.build.returns.read_ff3_csv` (divides by 100).
Why: dossier §7.5 asks for the Fama-French 3-factor model as a robustness variant of the abnormal returns (§11.4). `kpr.build.returns` implements the model (task Q2) but no frozen table carries the factors.
Who is affected: data (new collector, a few lines), quant (reads it in Q9 robustness runner), schemas.py + fixtures + test_schemas.py (contract PR).
Urgency: low. Only the robustness run of week S3 needs it; the main market model and the market-adjusted model need nothing new.
