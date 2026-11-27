"""Case study, Nvidia 2026: what the model predicted, what happened (dossier §12.2, §11.6).

Reads app/data/case_study.json (written by the export from the quant session's Q12 results):
{"title", "person_id", "company_id", "event_date", "story", "predicted_car", "actual_car",
 "demo"}.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from common import cached_case_study, load_all, page_header
from plotly.subplots import make_subplots

st.set_page_config(page_title="Case study · Key-Person Risk Radar", layout="wide")
page_header("Case study: Nvidia 2026", "An out-of-sample test on a real event.")

case = cached_case_study()
if case is None:
    st.info("This case study is being written. It will be published with the final results.")
    st.stop()

data = load_all()
st.subheader(case["title"])
st.markdown(case["story"])

c1, c2 = st.columns(2)
c1.metric("Predicted abnormal return J0 to J+2", f"{case['predicted_car']:.1%}")
c2.metric("Actual abnormal return J0 to J+2", f"{case['actual_car']:.1%}")

event_date = pd.Timestamp(case["event_date"])
daily = data["daily"]
around = daily[
    (daily["person_id"] == case["person_id"])
    & (daily["date"].between(event_date - pd.DateOffset(days=45),
                             event_date + pd.DateOffset(days=45)))
].sort_values("date")

if around.empty:
    st.info("No daily series around this date.")
    st.stop()

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_scatter(x=around["date"], y=around["close_adj"], name="Share price (adjusted)")
fig.add_bar(x=around["date"], y=around["views"], name="Wikipedia views", opacity=0.4,
            secondary_y=True)
fig.add_vline(x=event_date, line_dash="dash", line_color="grey")
fig.update_yaxes(title_text="Share price", secondary_y=False)
fig.update_yaxes(title_text="Wikipedia views", secondary_y=True)
st.plotly_chart(fig, width="stretch")
st.caption("The abnormal return is net of the market. Other news on the same days may "
           "contribute to it; the report discusses competing news.")
