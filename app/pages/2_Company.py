"""Company profile: tracked persons, dependence index, yearly history (dossier §12.2, §3.7)."""

from __future__ import annotations

import plotly.express as px
import streamlit as st
from common import company_dependence, fmt_score, fmt_v, load_all, page_header

st.set_page_config(page_title="Company · Key-Person Risk Radar", layout="wide")
data = load_all()
scores = data["scores_named"]

companies = (
    scores[["company_id", "company_name"]].drop_duplicates().sort_values("company_name")
)
names = dict(zip(companies["company_id"], companies["company_name"], strict=True))
company_id = st.sidebar.selectbox("Company", list(names), format_func=names.get)

page_header(names[company_id], "How much does this company depend on the people we track?")

rows = scores[scores["company_id"] == company_id].sort_values(["year", "person_name"])
dep = company_dependence(rows).sort_values("year")
last = dep.iloc[-1]

c1, c2, c3 = st.columns(3)
c1.metric(
    f"Dependence index ({last.year})",
    f"{last.dependence_index:.1f}% (estimate)",
    help="Largest V among the tracked persons: the share of market value tied to the most "
    "critical person.",
)
c2.metric(
    "Upper bound",
    f"{last.upper_bound:.1f}%",
    help="Sum of the V of all tracked persons. It can count one event twice when two "
    "persons are hit by the same news, so it is only an upper bound.",
)
c3.metric("Persons tracked", int(last.n_persons))

st.subheader("Tracked persons, latest year")
latest = rows[rows["year"] == rows["year"].max()]
for r in latest.itertuples():
    badge = " · :orange[low confidence]" if r.low_confidence else ""
    st.markdown(
        f"**{r.person_name}** ({r.role}{', founder' if r.founder else ''}): "
        f"S⁻ {fmt_score(r.s_minus)}, S⁺ {fmt_score(r.s_plus)}, "
        f"V {fmt_v(r.v_pct, r.v_lo, r.v_hi)}{badge}"
    )

st.subheader("Yearly scores")
metric = st.radio("Indicator", ["s_minus", "s_plus", "v_pct"], horizontal=True,
                  format_func={"s_minus": "S⁻", "s_plus": "S⁺", "v_pct": "V %"}.get)
fig = px.line(rows, x="year", y=metric, color="person_name", markers=True,
              labels={"year": "Year", metric: metric.replace("_", " "),
                      "person_name": "Person"})
for r in rows[rows["succession_flag"]].itertuples():
    fig.add_vline(x=r.year, line_dash="dash", line_color="grey")
    fig.add_annotation(x=r.year, y=1, yref="paper", text=f"succession: {r.person_name}",
                       showarrow=False, yanchor="bottom")
fig.update_xaxes(dtick=1)
st.plotly_chart(fig, width="stretch")
st.caption("Dashed lines mark a succession: the new person's score restarts from the "
           "average of their peer group, with no inheritance and no yearly cap (dossier §3.4).")
