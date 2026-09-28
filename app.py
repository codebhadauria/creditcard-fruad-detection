"""Interactive dashboard for credit-card anomaly detection."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import make_demo_data, validate_transactions
from src.model import fit_score


st.set_page_config(page_title="Card Fraud · Anomaly Lab", page_icon="💳", layout="wide")
st.title("💳 Credit Card Fraud Detection")
st.caption("Explore rare transactions with unsupervised anomaly detection. Scores are review signals, not fraud decisions.")

with st.sidebar:
    st.header("Data")
    source = st.radio("Choose a data source", ["Use built-in synthetic demo data", "Upload a CSV"])
    uploaded = None
    if source == "Upload a CSV":
        uploaded = st.file_uploader("Transaction CSV", type=["csv"], help="Optional Class column: 0 = legitimate, 1 = fraud")
    st.header("Detection settings")
    model_label = st.selectbox("Model", ["Isolation Forest", "Local Outlier Factor"])
    model_name = "isolation_forest" if model_label == "Isolation Forest" else "lof"
    contamination = st.number_input("Expected anomaly rate (%)", min_value=0.01, max_value=50.0,
                                    value=0.17, step=0.05, help="Sets the score threshold from the training data.") / 100
    run = st.button("Run detection", type="primary", use_container_width=True)

if source == "Use built-in synthetic demo data":
    data = make_demo_data()
    st.info("Showing generated synthetic transactions for demonstration. Use an appropriate real dataset for meaningful performance estimates.")
else:
    if uploaded is None:
        st.markdown("### Add a transaction CSV to begin")
        st.write("Expected format: one transaction per row, numeric feature columns, and optionally `Class` (`0` legitimate, `1` fraud).")
        st.stop()
    try:
        data = validate_transactions(pd.read_csv(uploaded))
    except Exception as exc:
        st.error(str(exc))
        st.stop()

with st.expander("Preview input data", expanded=False):
    st.write(f"{len(data):,} transactions · {len(data.columns) - int('Class' in data.columns)} numeric features")
    st.dataframe(data.head(10), use_container_width=True, hide_index=True)

if run:
    try:
        with st.spinner("Fitting detector and scoring the holdout…"):
            result = fit_score(data, model_name=model_name, contamination=contamination)
    except Exception as exc:
        st.error(f"Could not run detection: {exc}")
        st.stop()
    metrics = result["metrics"]
    scored = result["scored"]
    st.session_state["result"] = result
    st.session_state["model_name"] = model_label

result = st.session_state.get("result")
if result is None:
    st.markdown("### Ready to explore")
    st.write("Choose a model and select **Run detection**. The app trains on one part of the data and scores a held-out part.")
    st.stop()

metrics = result["metrics"]
scored = result["scored"]
st.subheader(f"Results · {st.session_state.get('model_name', metrics['model'])}")
if "precision" in metrics:
    cols = st.columns(5)
    for col, label, key in zip(cols, ["Precision", "Recall", "F1", "Average precision", "ROC AUC"],
                               ["precision", "recall", "f1", "average_precision", "roc_auc"]):
        col.metric(label, f"{metrics[key]:.3f}")
else:
    cols = st.columns(3)
    cols[0].metric("Transactions scored", f"{metrics['rows_scored']:,}")
    cols[1].metric("Review candidates", f"{metrics['anomalies_flagged']:,}")
    cols[2].metric("Threshold", f"{metrics['threshold']:.4f}")
    st.caption(metrics.get("evaluation_note", "No Class labels provided; precision/recall cannot be computed."))

left, right = st.columns([1, 1])
with left:
    st.markdown("#### Anomaly score distribution")
    if "Class" in scored.columns:
        plot = px.histogram(scored, x="anomaly_score", color=scored["Class"].map({0: "Legitimate", 1: "Fraud"}),
                            barmode="overlay", opacity=0.7, nbins=45,
                            labels={"color": "Known label", "anomaly_score": "Anomaly score"})
    else:
        plot = px.histogram(scored, x="anomaly_score", nbins=45)
    plot.add_vline(x=result["threshold"], line_dash="dash", line_color="#e74c3c", annotation_text="review threshold")
    st.plotly_chart(plot, use_container_width=True)
with right:
    st.markdown("#### Known labels vs detector")
    if "confusion_matrix" in metrics:
        cm = metrics["confusion_matrix"]
        heat = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                         labels={"x": "Predicted", "y": "Actual", "color": "Transactions"},
                         x=["Legitimate", "Fraud"], y=["Legitimate", "Fraud"])
        st.plotly_chart(heat, use_container_width=True)
    else:
        st.write("Add a `Class` column to calculate labeled evaluation metrics.")
        st.write(f"The detector flagged **{metrics['anomalies_flagged']:,}** of {metrics['rows_scored']:,} scored transactions.")

st.markdown("#### Highest scoring review candidates")
st.dataframe(scored.sort_values("anomaly_score", ascending=False).head(25), use_container_width=True, hide_index=True)
csv = scored.to_csv(index=False).encode("utf-8")
st.download_button("Download scored holdout CSV", data=csv, file_name="scored_transactions.csv", mime="text/csv")
st.caption(f"Trained with {metrics['rows_train']:,} rows; scored {metrics['rows_scored']:,}. Higher anomaly scores indicate more unusual patterns.")
