"""FraudLens main page: score transactions and list the flagged ones."""

import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_public_dataset, read_transactions_csv
from src.demo import find_column, load_demo
from src.explainability import explain_transaction, reason_to_text
from src.predict import load_bundle, score_transactions
from src.preprocessing import time_based_split

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="FraudLens", layout="wide")

DEMO = "Demo (simulated data)"
TEST = "Held-out test set"
UPLOAD = "Upload CSV"


@st.cache_resource
def get_bundle() -> dict:
    return load_bundle()


@st.cache_resource
def get_demo():
    return load_demo()


@st.cache_data
def get_test_set() -> pd.DataFrame:
    _, _, test = time_based_split(load_public_dataset())
    return test


st.title("FraudLens")
st.caption(
    "Flags potentially fraudulent transactions for analyst review. "
    "Risk scores rank transactions; they are not calibrated probabilities."
)

source = st.sidebar.radio("Data source", [DEMO, TEST, UPLOAD])

df = None
if source == DEMO:
    try:
        bundle, df = get_demo()
    except FileNotFoundError as err:
        st.error(str(err))
        st.stop()
    st.warning(
        "Demo mode: these transactions are SIMULATED, and the fraud patterns were "
        "planted by the generator. The model's results here show that the pipeline "
        "works, not that it detects real fraud."
    )
else:
    try:
        bundle = get_bundle()
    except FileNotFoundError as err:
        st.error(str(err))
        st.stop()
    if source == UPLOAD:
        uploaded = st.sidebar.file_uploader("Transactions CSV", type="csv")
        if uploaded is not None:
            try:
                df = read_transactions_csv(uploaded, bundle["features"])
            except ValueError as err:
                st.error(f"Cannot use this file: {err}")
                st.stop()
    else:
        try:
            df = get_test_set()
        except (FileNotFoundError, ValueError) as err:
            st.error(f"Could not load the public dataset: {err}")
            st.stop()

if df is None:
    st.info("Upload a CSV file to begin. It needs these columns: "
            + ", ".join(bundle["features"]))
    st.stop()

try:
    scored = score_transactions(df, bundle)
except ValueError as err:
    st.error(f"Could not score this file: {err}")
    st.stop()

label_col = find_column(scored, ["Class", "is_fraud"])
amount_col = find_column(scored, ["Amount", "amount"])

c1, c2, c3 = st.columns(3)
c1.metric("Transactions analyzed", f"{len(scored):,}")
c2.metric("Flagged for review (model)", f"{int(scored['flagged'].sum()):,}")
if label_col:
    c3.metric("Confirmed fraud (labels)", f"{int((scored[label_col] == 1).sum()):,}")
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
st.plotly_chart(fig, width="stretch")

st.subheader("Flagged transactions")
extra = ["Time", "timestamp", "Amount", "amount", "merchant_category",
         "Class", "is_fraud"]
show = ["risk_score", "risk_category"] + [c for c in extra if c in scored.columns]
flagged = scored[scored["flagged"]].sort_values("risk_score", ascending=False)

if flagged.empty:
    st.info("No transactions in this file were flagged for review.")
    st.stop()

st.dataframe(flagged[show], width="stretch")

st.subheader("Transaction details")
chosen = st.selectbox(
    "Select a flagged transaction",
    options=flagged.index.tolist(),
    format_func=lambda i: f"Row {i} (score {flagged.loc[i, 'risk_score']:.1f})",
)
row = scored.loc[chosen]

d1, d2, d3 = st.columns(3)
d1.metric("Risk score (0 to 100)", f"{row['risk_score']:.1f}")
d2.metric("Risk category", str(row["risk_category"]).title())
d3.metric("Amount", f"{row[amount_col]:,.2f}" if amount_col else "n/a")

if label_col:
    label = "Fraud" if int(row[label_col]) == 1 else "Not fraud"
    st.write(f"**Confirmed label in the data:** {label}")

st.subheader("Why was this flagged?")
try:
    reasons = explain_transaction(row, bundle)
except ValueError as err:
    st.warning(str(err))
else:
    if reasons:
        for reason in reasons:
            st.write("- " + reason_to_text(reason))
    else:
        st.write("No single feature stands out on its own for this transaction.")

st.caption(
    "The score ranks this transaction against others. It is not a probability of fraud. "
    "Reasons show how many score points disappear when one feature is set to its "
    "typical training value. They describe what the model used, not what causes "
    "fraud, and several features working together can be understated. "
    "In the real model V1 to V28 are anonymized, so they cannot be read as "
    "real-world attributes."
)