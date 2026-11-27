"""Leaderboard: full ranking, filterable by region, sector, role and indicator (§12.2)."""

from __future__ import annotations

import streamlit as st
from common import confidence_badge, fmt_v, load_all, page_header

st.set_page_config(page_title="Leaderboard · Key-Person Risk Radar", layout="wide")
page_header("Leaderboard", "Every tracked person, ranked. Click a column header to sort.")

data = load_all()
scores = data["scores_named"]

INDICATORS = {
    "V, implied share of market value": "v_pct",
    "S⁻, sensitivity to bad news (0–10)": "s_minus",
    "S⁺, sensitivity to good news (0–10)": "s_plus",
}

with st.sidebar:
    year = st.selectbox("Year", sorted(scores["year"].unique(), reverse=True))
    indicator = st.radio("Rank by", list(INDICATORS))
    regions = st.multiselect("Region", sorted(scores["region"].dropna().unique()))
    sectors = st.multiselect("Sector", sorted(scores["sector"].dropna().unique()))
    roles = st.multiselect("Role", sorted(scores["role"].dropna().unique()))

view = scores[scores["year"] == year]
for column, chosen in (("region", regions), ("sector", sectors), ("role", roles)):
    if chosen:
        view = view[view[column].isin(chosen)]
view = view.sort_values(INDICATORS[indicator], ascending=False).reset_index(drop=True)

if view.empty:
    st.info("No person matches these filters.")
    st.stop()

table = view.assign(
    rank=range(1, len(view) + 1),
    v_display=[fmt_v(r.v_pct, r.v_lo, r.v_hi) for r in view.itertuples()],
    confidence=[confidence_badge(n) for n in view["n_events"]],
)[["rank", "person_name", "company_name", "role", "region", "sector",
   "s_minus", "s_plus", "asymmetry", "v_pct", "v_display", "n_events", "confidence"]]

st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    column_config={
        "rank": st.column_config.NumberColumn("#", width="small"),
        "person_name": "Person",
        "company_name": "Company",
        "s_minus": st.column_config.NumberColumn("S⁻", format="%.1f"),
        "s_plus": st.column_config.NumberColumn("S⁺", format="%.1f"),
        "asymmetry": st.column_config.NumberColumn("S⁻/S⁺", format="%.1f"),
        "v_pct": st.column_config.NumberColumn("V % (sort)", format="%.1f"),
        "v_display": "V (estimate, 5–95% interval)",
        "n_events": st.column_config.NumberColumn("Shocks"),
        "confidence": "Confidence",
    },
)
st.caption("S⁻ and S⁺ are on a fixed 0–10 scale (same meaning every year). A 10 means the "
           "value reached the top of the scale. 'low confidence': fewer than 8 shocks.")
