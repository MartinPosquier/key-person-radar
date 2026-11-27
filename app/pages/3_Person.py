"""Person profile: S⁻, S⁺, asymmetry, V, and the list of shocks (dossier §12.2)."""

from __future__ import annotations

import plotly.express as px
import streamlit as st
from common import confidence_badge, fmt_score, fmt_usd, fmt_v, load_all, page_header

st.set_page_config(page_title="Person · Key-Person Risk Radar", layout="wide")
data = load_all()
scores = data["scores_named"]
events = data["events"]

persons = scores[["person_id", "person_name"]].drop_duplicates().sort_values("person_name")
names = dict(zip(persons["person_id"], persons["person_name"], strict=True))
default = list(names).index("jensen_huang") if "jensen_huang" in names else 0
person_id = st.sidebar.selectbox("Person", list(names), index=default, format_func=names.get)

rows = scores[scores["person_id"] == person_id].sort_values("year")
year = st.sidebar.selectbox("Year", rows["year"].tolist()[::-1])
r = rows[rows["year"] == year].iloc[0]

page_header(names[person_id], f"{r.role} of {r.company_name} · {year}")
badge = confidence_badge(int(r.n_events))
if badge:
    st.warning(f"Low confidence: this score rests on {int(r.n_events)} shocks (fewer than 8).")

c1, c2, c3, c4 = st.columns(4)
c1.metric("S⁻ bad news", fmt_score(r.s_minus), help=f"Interval {r.s_minus_lo:.1f}–"
          f"{r.s_minus_hi:.1f}. Raw: {r.s_minus_raw:.2f}% abnormal return.")
c2.metric("S⁺ good news", fmt_score(r.s_plus), help=f"Interval {r.s_plus_lo:.1f}–"
          f"{r.s_plus_hi:.1f}. Raw: {r.s_plus_raw:.2f}% abnormal return.")
c3.metric("Asymmetry S⁻/S⁺", "n/a" if r.asymmetry != r.asymmetry else f"{r.asymmetry:.1f}×")
c4.metric("V", fmt_v(r.v_pct, r.v_lo, r.v_hi), help=f"About {fmt_usd(r.v_usd)} (estimate).")

mine = events[(events["person_id"] == person_id) & ~events["excluded"]].sort_values("j0")
if mine.empty:
    st.info("No usable shock for this person.")
    st.stop()

st.subheader("Abnormal return against tone")
fig = px.scatter(
    mine, x="tone_rel", y="car_0_2", color="sign",
    color_discrete_map={"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"},
    hover_data={"j0": "|%Y-%m-%d", "z_wiki": ":.1f"},
    labels={"tone_rel": "Tone relative to this person's usual press",
            "car_0_2": "Abnormal return J0 to J+2"},
)
fig.update_yaxes(tickformat=".1%")
st.plotly_chart(fig, width="stretch")

st.subheader("Timeline of shocks")
fig = px.scatter(mine, x="j0", y="car_0_2", color="sign", size=mine["z_wiki"].clip(lower=1),
                 color_discrete_map={"negative": "#d62728", "neutral": "#7f7f7f",
                                     "positive": "#2ca02c"},
                 labels={"j0": "Date", "car_0_2": "Abnormal return J0 to J+2"})
fig.add_hline(y=0, line_color="grey")
fig.update_yaxes(tickformat=".1%")
st.plotly_chart(fig, width="stretch")

st.subheader("Shocks")
st.dataframe(
    mine[["j0", "sign", "tone_rel", "z_wiki", "car_0_2", "car_pred", "split"]],
    hide_index=True,
    width="stretch",
    column_config={
        "j0": st.column_config.DateColumn("Date (J0)"),
        "tone_rel": st.column_config.NumberColumn("Tone", format="%.2f"),
        "z_wiki": st.column_config.NumberColumn("Attention z", format="%.1f"),
        "car_0_2": st.column_config.NumberColumn("Actual abnormal return", format="%.4f"),
        "car_pred": st.column_config.NumberColumn("Predicted", format="%.4f"),
    },
)
st.caption(f"Abnormal return of {r.company_name}'s stock around each event, net of the "
           "market. It describes what the stock did, not what the person caused.")
