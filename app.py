"""FraudLens: first working page. Score transactions and show the flagged ones."""

import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_public_dataset
from src.predict import load_bundle, score_transactions
from src.preprocessing import time_based_split

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="FraudLens", layout="wide")


@st.cache_resource
def get_bundle() -> dict:
    return load_bundle()


@st.cache_data
def get_test_set() -> pd.DataFrame:
    _, _, test = time_based_split(load_public_dataset())
    return test


st.title("FraudLens")
st.caption(
    "Flags potentially fraudulent transactions for analyst review. "
    "Risk scores rank transactions; they are not calibrated probabilities."
)

try:
    bundle = get_bundle()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

source = st.sidebar.radio("Data source", ["Held-out test set", "Upload CSV"])

df = None
if source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Transactions CSV", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
else:
    try:
        df = get_test_set()
    except (FileNotFoundError, ValueError) as err:
        st.error(f"Could not load the public dataset: {err}")
        st.stop()

if df is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

try:
    scored = score_transactions(df, bundle)
except ValueError as err:
    st.error(f"Could not score this file: {err}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Transactions analyzed", f"{len(scored):,}")
c2.metric("Flagged for review (model)", f"{int(scored['flagged'].sum()):,}")
if "Class" in scored.columns:
    c3.metric("Confirmed fraud (labels)", f"{int((scored['Class'] == 1).sum()):,}")
else:
    c3.metric("Confirmed fraud (labels)", "n/a")

order = ["low", "medium", "high"]
counts = scored["risk_category"].value_counts().reindex(order, fill_value=0)
fig = px.bar(
    x=order,
    y=counts.to_list(),
    log_y=True,
    labels={"x": "Risk category", "y": "Transactions (log scale)"},
    title="Transactions by risk category",
)
fig.update_yaxes(dtick=1)
fig.update_traces(texttemplate="%{y:,}", textposition="outside")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Flagged transactions")
show = ["risk_score", "risk_category"] + [c for c in ("Time", "Amount", "Class") if c in scored.columns]
flagged = scored[scored["flagged"]].sort_values("risk_score", ascending=False)
st.dataframe(flagged[show], use_container_width=True)