"""Transaction Explorer: load transactions, filter, inspect and download."""

import pandas as pd
import streamlit as st

from src.data_loader import load_public_dataset, read_transactions_csv
from src.demo import find_column, load_demo
from src.explainability import explain_transaction, reason_to_text
from src.predict import load_bundle, score_transactions
from src.preprocessing import time_based_split

MAX_TABLE_ROWS = 2000
DEMO = "Demo (simulated data)"
TEST = "Held-out test set"
UPLOAD = "Upload CSV"

st.set_page_config(page_title="Transaction Explorer", layout="wide")


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


st.title("Transaction Explorer")
st.caption("Filter scored transactions, inspect one, and download the results.")

source = st.sidebar.radio("Data source", [DEMO, TEST, UPLOAD])

df = None
if source == DEMO:
    try:
        bundle, df = get_demo()
    except FileNotFoundError as err:
        st.error(str(err))
        st.stop()
    st.warning(
        "Demo mode: these transactions are SIMULATED, with fraud patterns planted "
        "by the generator. Results show that the pipeline works, not that it "
        "detects real fraud."
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

scored = score_transactions(df, bundle)
label_col = find_column(scored, ["Class", "is_fraud"])

st.sidebar.header("Filters")
categories = st.sidebar.multiselect(
    "Risk category", ["low", "medium", "high"], default=["low", "medium", "high"]
)
flagged_only = st.sidebar.checkbox("Flagged only", value=True)
score_low, score_high = st.sidebar.slider("Risk score range", 0, 100, (0, 100))

mask = (scored["risk_category"].isin(categories)
        & scored["risk_score"].between(score_low, score_high))
if flagged_only:
    mask &= scored["flagged"]

if "timestamp" in scored.columns:
    dates = pd.to_datetime(scored["timestamp"]).dt.date
    first, last = dates.min(), dates.max()
    picked = st.sidebar.date_input(
        "Date range", (first, last), min_value=first, max_value=last
    )
    if len(picked) == 2:
        mask &= dates.between(picked[0], picked[1])

view = scored[mask].sort_values("risk_score", ascending=False)

st.write(f"Showing **{len(view):,}** of {len(scored):,} transactions")
if view.empty:
    st.info("No transactions match these filters.")
    st.stop()

extra = ["Time", "timestamp", "Amount", "amount", "merchant_category",
         "Class", "is_fraud"]
columns = ["risk_score", "risk_category", "flagged"] + [
    c for c in extra if c in view.columns
]
st.dataframe(view[columns].head(MAX_TABLE_ROWS), width="stretch")
if len(view) > MAX_TABLE_ROWS:
    st.caption(f"Table shows the top {MAX_TABLE_ROWS:,} by risk score. "
               "The download contains all matching rows.")

st.download_button(
    "Download filtered results (CSV)",
    data=view.to_csv(index=False).encode("utf-8"),
    file_name="fraudlens_filtered.csv",
    mime="text/csv",
)

st.subheader("Inspect one transaction")
chosen = st.selectbox(
    "Select a transaction",
    options=view.head(MAX_TABLE_ROWS).index.tolist(),
    format_func=lambda i: f"Row {i} (score {view.loc[i, 'risk_score']:.1f})",
)
row = view.loc[chosen]

m1, m2 = st.columns(2)
m1.metric("Risk score (0 to 100)", f"{row['risk_score']:.1f}")
m2.metric("Risk category", str(row["risk_category"]).title())
if label_col:
    label = "Fraud" if int(row[label_col]) == 1 else "Not fraud"
    st.write(f"**Confirmed label in the data:** {label}")

try:
    reasons = explain_transaction(row, bundle)
except ValueError as err:
    st.warning(str(err))
else:
    st.write("**Why this score?**")
    if reasons:
        for reason in reasons:
            st.write("- " + reason_to_text(reason))
    else:
        st.write("No single feature stands out on its own.")

st.caption("Scores rank transactions; they are not probabilities of fraud.")