"""Methodology: plain explanation, sources, limits, disclaimer (dossier §12.2, §12.4)."""

from __future__ import annotations

import streamlit as st
from common import cached_meta, page_header

st.set_page_config(page_title="Methodology · Key-Person Risk Radar", layout="wide")
page_header("Methodology", "How the scores are built, and what they cannot tell you.")

st.markdown(
    """
### In five lines

1. An **abnormal return** is what a stock did beyond what the market as a whole did that
   day: if the market rose 1% and the stock fell 2%, the abnormal return is about -3%.
2. We detect days when the public suddenly pays attention to a person (Wikipedia views and
   press coverage far above their usual level), and measure the tone of that coverage.
3. We **remove earnings days**, because results move the stock for reasons that have
   nothing to do with the person.
4. A model learns how the stock's abnormal return over three trading days responds to
   these shocks, depending on their tone and context.
5. From that model we derive S⁻ and S⁺ (reaction to a standard bad or good piece of news,
   on a fixed 0–10 scale) and V (share of market value at risk on an abrupt departure).
"""
)

st.subheader("Pipeline")
st.graphviz_chart(
    """
digraph {
  rankdir=LR; node [shape=box, style=rounded];
  wiki [label="Wikipedia views"]; news [label="Press (GDELT)"];
  prices [label="Prices, indices, VIX"]; earn [label="Earnings dates"];
  panel [label="Daily panel"]; events [label="Attention shocks\\n+ abnormal returns"];
  tone [label="Tone (FinBERT)"]; model [label="ML models"];
  scores [label="S⁻, S⁺, V"]; app [label="This app"];
  wiki -> panel; news -> panel; prices -> panel; earn -> panel;
  panel -> events; tone -> model; events -> model; model -> scores; scores -> app;
}
"""
)

st.subheader("Sources")
st.markdown(
    """
- Wikipedia pageviews (Wikimedia REST API), English and local language.
- GDELT news coverage and tone.
- Daily prices and indices (Yahoo Finance), VIX.
- Earnings dates from SEC EDGAR filings (US) and company calendars (Europe).
- Mandates (who held which role, when) from Wikidata, checked by hand.
"""
)

st.subheader("Limits")
st.markdown(
    """
- **Association, not causation.** A stock moving on the day a person is in the news does
  not prove the person caused the move; other news may coincide.
- Every V is an **estimate** with a wide interval, shown as the 5th–95th bootstrap
  percentiles. Scores based on fewer than 8 shocks are marked *low confidence*.
- Abrupt departures are rare, so the calibration of V rests on few cases.
- Only persons with an English Wikipedia page and regular press coverage can be measured.
"""
)

meta = cached_meta()
if meta:
    st.caption(f"Data origin: {meta.get('source', 'unknown')} · exported "
               f"{meta.get('exported_at', 'unknown')}.")
st.caption("Research project by ESILV students. Not investment advice.")
