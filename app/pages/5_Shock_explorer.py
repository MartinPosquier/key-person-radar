"""Shock explorer: every shock, filterable, and the mean CAR curve J-10 to J+10 (§12.2)."""

from __future__ import annotations

import plotly.express as px
import streamlit as st
from common import load_all, mean_car_curve, page_header

st.set_page_config(page_title="Shock explorer · Key-Person Risk Radar", layout="wide")
page_header("Shock explorer", "Every attention shock we detected, and how stocks moved.")

data = load_all()
names = data["entities"].drop_duplicates("person_id").set_index("person_id")["person_name"]
events = data["events"][~data["events"]["excluded"]].copy()
events["person_name"] = events["person_id"].map(names).fillna(events["person_id"])

with st.sidebar:
    signs = st.multiselect("Type of news", ["negative", "neutral", "positive"],
                           default=["negative", "neutral", "positive"])
    persons = st.multiselect("Person", sorted(events["person_name"].unique()))
view = events[events["sign"].isin(signs)]
if persons:
    view = view[view["person_name"].isin(persons)]

st.subheader("Mean cumulative abnormal return around the shock")
curve = mean_car_curve(view, data["daily"])
if curve.empty:
    st.info("Not enough daily data around the selected shocks.")
else:
    fig = px.line(
        curve, x="k", y="car", color="sign", markers=True, hover_data={"n": True},
        color_discrete_map={"negative": "#d62728", "neutral": "#7f7f7f",
                            "positive": "#2ca02c"},
        labels={"k": "Trading days relative to the shock (J0)",
                "car": "Mean cumulative abnormal return", "n": "shocks"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="grey")
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, width="stretch")
    st.caption("Cumulated from J-10. A drift before J0 would signal that the news leaked or "
               "was anticipated; the reaction window used by the model is J0 to J+2.")

st.subheader(f"{len(view):,} shocks")
st.dataframe(
    view.sort_values("j0", ascending=False)[
        ["j0", "person_name", "company_id", "sign", "tone_rel", "z_wiki", "car_0_2"]
    ],
    hide_index=True,
    width="stretch",
    column_config={
        "j0": st.column_config.DateColumn("Date (J0)"),
        "person_name": "Person",
        "company_id": "Company",
        "tone_rel": st.column_config.NumberColumn("Tone", format="%.2f"),
        "z_wiki": st.column_config.NumberColumn("Attention z", format="%.1f"),
        "car_0_2": st.column_config.NumberColumn("Abnormal return J0–J+2", format="%.4f"),
    },
)
