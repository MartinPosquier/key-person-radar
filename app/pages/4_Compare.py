"""Compare two persons side by side, Huang vs Pouyanné by default (dossier §12.2)."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from common import fmt_v, latest_year, load_all, page_header

st.set_page_config(page_title="Compare · Key-Person Risk Radar", layout="wide")
page_header("Compare", "Two persons side by side, with their uncertainty.")

data = load_all()
latest = latest_year(data["scores_named"]).sort_values("v_pct", ascending=False)
names = dict(zip(latest["person_id"], latest["person_name"], strict=True))
ids = list(names)

if len(ids) < 2:
    st.info("At least two persons are needed.")
    st.stop()

defaults = [p for p in ("jensen_huang", "patrick_pouyanne") if p in names]
defaults += [p for p in ids if p not in defaults][: 2 - len(defaults)]
col_a, col_b = st.columns(2)
a = col_a.selectbox("Person A", ids, index=ids.index(defaults[0]), format_func=names.get)
b = col_b.selectbox("Person B", ids, index=ids.index(defaults[1]), format_func=names.get)
pair = latest.set_index("person_id").loc[[a, b]]

fig = go.Figure()
for label, col, lo, hi in (("S⁻", "s_minus", "s_minus_lo", "s_minus_hi"),
                           ("S⁺", "s_plus", "s_plus_lo", "s_plus_hi")):
    fig.add_bar(
        name=label, x=pair["person_name"], y=pair[col],
        error_y={"type": "data", "symmetric": False,
                 "array": (pair[hi] - pair[col]).clip(lower=0),
                 "arrayminus": (pair[col] - pair[lo]).clip(lower=0)},
    )
fig.update_layout(barmode="group", yaxis={"range": [0, 10.5], "title": "Score (0–10)"},
                  height=380)
st.plotly_chart(fig, width="stretch")

for col, (_pid, r) in zip(st.columns(2), pair.iterrows(), strict=True):
    with col:
        st.markdown(f"**{r.person_name}**, {r.role} of {r.company_name} ({r.year})")
        st.markdown(f"V: {fmt_v(r.v_pct, r.v_lo, r.v_hi)}")
        st.markdown(f"Shocks: {int(r.n_events)}"
                    + (" · :orange[low confidence]" if r.low_confidence else ""))

overlap = not (pair.iloc[0].s_minus_hi < pair.iloc[1].s_minus_lo
               or pair.iloc[1].s_minus_hi < pair.iloc[0].s_minus_lo)
if overlap:
    st.caption("The S⁻ intervals overlap: the data does not separate these two persons on "
               "sensitivity to bad news.")
