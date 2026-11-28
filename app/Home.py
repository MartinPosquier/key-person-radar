"""Home page: the hook, 3 key figures, top 10 persons by V (dossier §12.1, §12.2).

Run: streamlit run app/Home.py
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from common import fmt_usd, fmt_v, latest_year, load_all, page_header

st.set_page_config(page_title="Key-Person Risk Radar", page_icon="📡", layout="wide")

data = load_all()
latest = latest_year(data["scores_named"]).sort_values("v_pct", ascending=False)

# The hook names Nvidia and Jensen Huang when they are in the data (dossier §12.1).
hero = latest[latest["person_id"] == "jensen_huang"]
if hero.empty:
    hero = latest.head(1)
top = hero.iloc[0]

page_header(
    f"How much of {top.company_name} is {top.person_name}?",
    f"We measured how much {data['entities']['company_id'].nunique()} of the world's largest "
    "companies depend on a handful of people.",
)

st.markdown(
    f"### {top.person_name}: {fmt_v(top.v_pct, top.v_lo, top.v_hi)} of "
    f"{top.company_name}'s market value at risk"
)
st.caption(
    "Share of market value that would be at risk on an abrupt departure, implied by how the "
    "stock reacted to past news about this person. Interval: 5th–95th bootstrap percentiles."
)

c1, c2, c3 = st.columns(3)
c1.metric("Persons tracked", f"{latest['person_id'].nunique()}")
c2.metric("Shocks analysed", f"{len(data['events'].query('not excluded')):,}")
c3.metric(
    "Value at risk, top 10 combined",
    fmt_usd(latest.head(10)["v_usd"].sum()),
    help="Sum of the top 10 estimates. Point estimates only: see each person's interval.",
)

st.subheader("Top 10 persons by implied share of market value (V)")
top10 = latest.head(10).iloc[::-1]
labels = [
    f"{r.person_name} ({r.company_name})" + (" ·low confidence" if r.low_confidence else "")
    for r in top10.itertuples()
]
fig = go.Figure(
    go.Bar(
        x=top10["v_pct"],
        y=labels,
        orientation="h",
        error_x={
            "type": "data",
            "symmetric": False,
            "array": (top10["v_hi"] - top10["v_pct"]).clip(lower=0),
            "arrayminus": (top10["v_pct"] - top10["v_lo"]).clip(lower=0),
        },
        hovertemplate="%{y}<br>V = %{x:.1f}% (estimate)<extra></extra>",
    )
)
fig.update_layout(
    xaxis_title="V, % of market value (estimate, 5th–95th percentile interval)",
    height=420,
    margin={"l": 10, "r": 10, "t": 10, "b": 10},
)
st.plotly_chart(fig, width="stretch")
st.caption(f"Latest available year per person. Scores on fewer than 8 shocks are marked "
           f"'low confidence'. Data: {data['scores']['year'].min()}–"
           f"{data['scores']['year'].max()}.")
