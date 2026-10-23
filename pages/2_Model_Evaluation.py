"""Model Evaluation: offline metrics from held-out data."""

import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import config

st.set_page_config(page_title="Model Evaluation", layout="wide")
st.title("Model Evaluation")
st.caption("Offline results on held-out data. These are not live performance numbers.")


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None


metrics = load_json(config.METRICS_PATH)
comparison = load_json(config.COMPARISON_PATH)
if metrics is None:
    st.error("models/metrics.json not found. Train the model first: "
             "python -m src.train_model")
    st.stop()

test = metrics["test"]
n_test = metrics["rows"]["test"]
frauds_test = metrics["frauds"]["test"]
tp, fp, fn = test["true_positives"], test["false_positives"], test["false_negatives"]
tn = n_test - tp - fp - fn

# --- Final model on the held-out test set --------------------------------
st.header("Final model on the held-out test set")
st.write(
    f"Random Forest, flag threshold {metrics['threshold']}. The test set has "
    f"{n_test:,} transactions, of which {frauds_test} are fraud. It was used once, "
    "after the model and threshold were chosen on validation data."
)

cols = st.columns(5)
cols[0].metric("Precision", f"{test['precision']:.3f}")
cols[1].metric("Recall", f"{test['recall']:.3f}")
cols[2].metric("F1-score", f"{test['f1']:.3f}")
cols[3].metric("PR-AUC", f"{test['pr_auc']:.3f}")
cols[4].metric("ROC-AUC", f"{test['roc_auc']:.3f}")

cols = st.columns(3)
cols[0].metric("Frauds caught", f"{tp}")
cols[1].metric("Frauds missed (false negatives)", f"{fn}")
cols[2].metric("False alarms (false positives)", f"{fp}")

counts = [[tn, fp], [fn, tp]]
fig = px.imshow(
    np.log10(np.array(counts) + 1),
    x=["Predicted genuine", "Predicted fraud"],
    y=["Actually genuine", "Actually fraud"],
    color_continuous_scale="Blues",
    title="Confusion matrix (color on a log scale; numbers are real counts)",
)
fig.update_traces(text=counts, texttemplate="%{text:,}")
fig.update_coloraxes(showscale=False)
st.plotly_chart(fig, width="stretch")

# --- Why accuracy is not enough -------------------------------------------
st.header("Why accuracy alone is not enough")
accuracy = (tn + tp) / n_test
do_nothing = 1 - frauds_test / n_test
a, b = st.columns(2)
a.metric("Accuracy of this model", f"{accuracy:.2%}")
b.metric("Accuracy of a model that flags nothing", f"{do_nothing:.2%}")
st.write(
    "Fraud is so rare that a model that never flags anything looks almost perfect "
    "by accuracy while catching zero fraud. That is why this project uses "
    "precision (how many alerts were real), recall (how many frauds were caught) "
    "and PR-AUC (how well fraud is ranked above genuine transactions)."
)

# --- Candidate model comparison -------------------------------------------
st.header("Candidate models (validation set)")
if comparison is None:
    st.info("models/model_comparison.json not found.")
else:
    table = pd.DataFrame(comparison).T
    table["alerts_to_review"] = table["true_positives"] + table["false_positives"]
    table.index = [name.replace("_", " ").title() for name in table.index]
    shown = table[["threshold", "precision", "recall", "f1", "pr_auc", "roc_auc",
                   "false_positives", "false_negatives", "alerts_to_review"]]
    st.dataframe(shown.round(3), width="stretch")
    st.write(
        f"The validation set has only {metrics['frauds']['validation']} frauds, so "
        "the gap between the two models' PR-AUC is within noise. The Random Forest "
        "was chosen because it reaches a similar number of caught frauds with far "
        "fewer alerts for analysts to review. Model and threshold were chosen on "
        "this table, never on the test set."
    )

# --- Threshold and limitations --------------------------------------------
with st.expander("How the threshold is chosen"):
    st.write(
        "A transaction is flagged when its risk score is at or above the threshold. "
        "Raising it gives fewer false alarms but more missed fraud; lowering it does "
        "the reverse. The right value depends on the cost of a missed fraud, the "
        "cost of a false alarm, and how many alerts analysts can review. In this "
        "project the costs were assumed values (100 per missed fraud, 2 per false "
        "alarm), so real deployments would use the organisation's own figures."
    )

st.header("Limitations")
st.write(
    f"- The test set contains only {frauds_test} frauds, so these metrics are noisy.\n"
    "- Features V1 to V28 are anonymized, so results cannot be tied to real-world "
    "attributes.\n"
    "- The data covers two days in September 2013. Performance on other periods or "
    "other institutions is unknown.\n"
    "- Risk scores are rankings, not calibrated probabilities.\n"
    "- These are offline results. Live performance would need monitoring."
)