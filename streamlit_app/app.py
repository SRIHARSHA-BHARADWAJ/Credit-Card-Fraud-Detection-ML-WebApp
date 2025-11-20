import streamlit as st
import requests
import numpy as np
import pandas as pd
import time
import math

from utils_plots import (
    plot_roc_curve,
    plot_precision_recall,
    plot_confusion_matrix
)

# ===================================================
# GLOBAL VARIABLES (must be before functions)
# ===================================================
y_true_global = None
out_df_global = None

# ===================================================
# CONFIGURE PAGE
# ===================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

# ===================================================
# API URLS
# ===================================================
API_SINGLE = "https://credit-card-fraud-detection-ml-webapp.onrender.com/predict"
API_BATCH = "https://credit-card-fraud-detection-ml-webapp.onrender.com/predict-batch"
API_MODELS = "https://credit-card-fraud-detection-ml-webapp.onrender.com/get-models"

# ===================================================
# PREMIUM UI STYLING
# ===================================================
st.markdown("""
<style>
    .main-title { font-size: 40px; font-weight: 700; text-align: center; color: #0E1117; }
    .sub-text { font-size: 18px; text-align: center; color: #4F4F4F; }
    .result-card { padding: 20px; border-radius: 15px; background: #f5f7fa; border: 1px solid #e2e2e2; text-align: center; margin-top: 20px; }
    .probability-box { font-size: 26px; font-weight: 700; color: #0077ff; }
</style>
""", unsafe_allow_html=True)

# ===================================================
# HEADER
# ===================================================
st.markdown("<h1 class='main-title'>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Select a model, enter values or upload a CSV file to detect fraud instantly.</p>", unsafe_allow_html=True)
st.write("")

# ===================================================
# SIDEBAR
# ===================================================
st.sidebar.header("⚙️ Choose Model")

try:
    models = requests.get(API_MODELS).json().get("available_models", ["logreg", "rf"])
except:
    models = ["logreg", "rf"]

model = st.sidebar.radio("Select a Machine Learning Model:", models, index=0)

mode = st.sidebar.selectbox(
    "Choose Input Method:",
    ["Manual Input (5-6 values)", "Upload CSV File (FAST MODE)"]
)

# ===================================================
# API CALL (SINGLE PREDICTION)
# ===================================================
def call_api(features_list, model_selected):
    payload = {"features": features_list}
    try:
        resp = requests.post(f"{API_SINGLE}?model={model_selected}", json=payload)
        return resp.json(), resp.status_code
    except:
        return {"error": "Server unreachable"}, 500

# ===================================================
# API CALL (BATCH PREDICTION)
# ===================================================
def predict_in_chunks(df, model_name="rf", chunk_size=4000):
    global y_true_global
    global out_df_global

    n = len(df)
    chunks = math.ceil(n / chunk_size)

    preds = []
    probs = []

    st.info(f"Processing {n:,} rows in {chunks} chunks (~{chunk_size} rows/chunk).")
    progress = st.progress(0)
    status = st.empty()

    start_time = time.time()

    for i in range(chunks):
        s = i * chunk_size
        e = min((i + 1) * chunk_size, n)

        batch = df.iloc[s:e].values.tolist()
        payload = {"features": batch}

        # backend call
        try:
            r = requests.post(
                f"{API_BATCH}?model={model_name}",
                json=payload,
                timeout=300
            )
        except Exception as err:
            st.error(f"Network/timeout error at chunk {i+1}: {err}")
            return None

        try:
            out = r.json()
        except:
            st.error(f"Backend returned non-JSON at chunk {i+1}: {r.text}")
            return None

        if "predictions" not in out:
            st.error(f"Backend error at chunk {i+1}: {out}")
            return None

        preds.extend(out["predictions"])
        probs.extend(out["probabilities"])

        progress.progress(e / n)

        elapsed = time.time() - start_time
        eta = (elapsed / e) * (n - e) if e > 0 else 0
        status.text(f"Chunk {i+1}/{chunks} — {e}/{n} rows — ETA {eta/60:.2f} mins")

    # attach predictions
    df["prediction"] = preds
    df["fraud_probability"] = probs

    # attach true labels only if exist
    if y_true_global is not None:
        df["true_label"] = y_true_global.iloc[:len(df)]

    out_df_global = df
    return df

# ===================================================
# MANUAL INPUT MODE
# ===================================================
if mode == "Manual Input (5-6 values)":
    st.subheader("🧮 Manual Input Mode")

    col1, col2 = st.columns(2)
    with col1:
        f1 = st.number_input("Feature 1", 0.0)
        f2 = st.number_input("Feature 2", 0.0)
        f3 = st.number_input("Feature 3", 0.0)
    with col2:
        f4 = st.number_input("Feature 4", 0.0)
        f5 = st.number_input("Feature 5", 0.0)
        f6 = st.number_input("Feature 6", 0.0)

    if st.button("🚀 Predict Fraud"):
        features = [f1, f2, f3, f4, f5, f6] + [0.0] * 24
        result, status = call_api(features, model)

        if status == 200:
            pred = result["prediction"]
            prob = result["fraud_probability"]

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("🔍 Prediction Result")

            if pred == 1:
                st.error("⚠️ Fraud Detected!")
            else:
                st.success("✅ Legitimate Transaction")

            st.markdown(f"<p class='probability-box'>Fraud Probability: {prob}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ===================================================
# CSV UPLOAD MODE
# ===================================================
if mode == "Upload CSV File (FAST MODE)":
    st.subheader("📂 Upload CSV File (FAST MODE)")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # store true labels
        
        y_true_global = df["Class"].copy() if "Class" in df.columns else None

        df = df.drop(columns=["Class"], errors="ignore").iloc[:, :30]

        st.write("### Preview:")
        st.dataframe(df.head())

        if st.button("🚀 Predict for All Rows"):
            out_df = predict_in_chunks(df, model_name=model)

            if out_df is None:
                st.error("Batch prediction failed — check backend logs.")
                st.stop()

            st.success("Batch Prediction Complete!")
            st.dataframe(out_df.head())

            csv = out_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions CSV", csv, "predictions.csv", "text/csv")

# ===================================================
# VISUALIZATION BLOCK
# ===================================================
st.subheader("📊 Model Performance Visualizations")

if y_true_global is not None and out_df_global is not None:
    try:
        plot_roc_curve(y_true_global, out_df_global["fraud_probability"])
        plot_precision_recall(y_true_global, out_df_global["fraud_probability"])
        plot_confusion_matrix(y_true_global, out_df_global["prediction"])
    except:
        st.warning("Visualization failed due to missing or invalid values.")
else:
    st.info("📌 Upload a CSV containing 'Class' column to view performance visualizations.")
