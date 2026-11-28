From: data
To: quant
Need: none, information only
Why: `prices.ret` and `market.ret` (task D3) are log returns of the adjusted close, as asked
by dossier §6.4. The fixtures use simple returns. Sum them for CARs; convert with
`np.expm1` if a simple return is needed. Details: decisions/data/2026-11-27-prices.md.
Urgency: low
