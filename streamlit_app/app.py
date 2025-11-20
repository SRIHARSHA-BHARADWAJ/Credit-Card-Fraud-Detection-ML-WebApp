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
# PAGE CONFIG
# ===================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

# ===================================================
# SESSION STATE (REPLACES global)
# ===================================================
if "y_true" not in st.session_state:
    st.session_state.y_true = None

if "out_df" not in st.session_state:
    st.session_state.out_df = None

# ===================================================
# API ROUTES
# ===================================================
API_SINGLE = "https://credit-card-fraud-detection-ml-webapp.onrender.com/predict"
API_BATCH = "https://credit-card-fraud-detection-ml-webapp.onrender.com/predict-batch"
API_MODELS = "https://credit-card-fraud-detection-ml-webapp.onrender.com/get-models"

# ===================================================
# STYLING
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
st.markdown("<p class='sub-text'>Select a model, enter values or upload a CSV to detect fraud instantly.</p>", unsafe_allow_html=True)
st.write("")

# ===================================================
# SIDEBAR
# ===================================================
st.sidebar.header("⚙️ Choose Model")

try:
    models = requests.get(API_MODELS, timeout=5).json().get("available_models", ["logreg", "rf"])
except Exception:
    models = ["logreg", "rf"]

model = st.sidebar.radio("Select a Machine Learning Model:", models, index=0)

mode = st.sidebar.selectbox(
    "Choose Input Method:",
    ["Manual Input (6 values, rest zeros)", "Upload CSV File (FAST MODE)"]
)

# ===================================================
# FEATURE ORDER HANDLING
# ===================================================
# We'll default to Kaggle-style order: V1..V28, Amount, Time
KAGGLE_ORDER = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]


def prepare_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with exactly 30 features in the order expected by the backend.

    Strategy:
    1. If df already contains named columns V1..V28/Amount/Time in any order, pick them and reorder to KAGGLE_ORDER.
    2. If df has no V* columns but has 30 numeric columns (after dropping Class), take the first 30 numeric columns.
    3. If fewer than 30 columns exist, pad with zeros (not ideal, but keeps pipeline working).
    """
    df_local = df.copy()

    # Drop Class if present
    if "Class" in df_local.columns:
        df_local = df_local.drop(columns=["Class"])

    # Reset index to avoid alignment issues
    df_local = df_local.reset_index(drop=True)

    # Case 1: contains V1..V28 or Amount/Time
    present = [c for c in KAGGLE_ORDER if c in df_local.columns]
    if len(present) >= 2:
        # build columns list that are available, fill missing with zeros
        cols = []
        for c in KAGGLE_ORDER:
            if c in df_local.columns:
                cols.append(c)
            else:
                # create zero column for missing
                df_local[c] = 0.0
                cols.append(c)
        feat_df = df_local[cols]
        return feat_df

    # Case 2: no V* columns but many numeric columns
    numeric_cols = df_local.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) >= 30:
        feat_df = df_local[numeric_cols[:30]]
        feat_df.columns = KAGGLE_ORDER  # rename to expected names
        return feat_df

    # Case 3: fewer columns -> take what we have, then pad
    feat_df = df_local.copy()
    needed = 30 - feat_df.shape[1]
    for i in range(needed):
        feat_df[f"pad_{i}"] = 0.0

    # Ensure 30 columns
    feat_df = feat_df.iloc[:, :30]
    feat_df.columns = KAGGLE_ORDER
    return feat_df


# ===================================================
# SINGLE PREDICTION
# ===================================================
def call_api(features_list, model_selected):
    payload = {"features": features_list}
    try:
        resp = requests.post(f"{API_SINGLE}?model={model_selected}", json=payload, timeout=10)
        return resp.json(), resp.status_code
    except Exception as e:
        return {"error": f"Server unreachable: {e}"}, 500


# ===================================================
# BATCH PREDICTION
# ===================================================
def predict_in_chunks(df, model_name="rf", chunk_size=4000):

    # prepare features (reorder / pad)
    feat_df = prepare_features_df(df)

    n = len(feat_df)
    chunks = math.ceil(n / chunk_size) if n > 0 else 0

    preds = []
    probs = []

    st.info(f"Processing {n:,} rows in {chunks} chunks (~{chunk_size} rows/chunk).")
    progress = st.progress(0)
    status = st.empty()
    start_time = time.time()

    for i in range(chunks):
        s = i * chunk_size
        e = min((i + 1) * chunk_size, n)

        batch = feat_df.iloc[s:e].values.tolist()
        payload = {"features": batch}

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
        except Exception:
            st.error(f"Backend returned non-JSON at chunk {i+1}: {r.text}")
            return None

        if "predictions" not in out:
            st.error(f"Backend error at chunk {i+1}: {out}")
            return None

        preds.extend(out["predictions"])
        probs.extend(out.get("probabilities", [None] * (e - s)))

        progress.progress(e / n if n else 1.0)

        elapsed = time.time() - start_time
        eta = (elapsed / e) * (n - e) if e > 0 else 0
        status.text(f"Chunk {i+1}/{chunks} — {e}/{n} rows — ETA {eta/60:.2f} mins")

    # build output df: take original df (reset index) and append cols
    out_df = df.reset_index(drop=True).copy()
    out_df = out_df.iloc[:n]  # ensure length matches features

    out_df["prediction"] = preds
    out_df["fraud_probability"] = probs

    # attach true labels (reset index to align)
    if st.session_state.y_true is not None:
        y = st.session_state.y_true.reset_index(drop=True)
        out_df["true_label"] = y.iloc[:len(out_df)].values

    # save for visualization
    st.session_state.out_df = out_df
    return out_df


# ===================================================
# MANUAL MODE
# ===================================================
if mode == "Manual Input (6 values, rest zeros)":
    st.subheader("🧮 Manual Input Mode")

    # Keep the original lightweight manual input (6 values) and pad rest
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
        # build a 30-length feature vector (KAGGLE_ORDER). We'll place the 6 manual inputs in the first 6 V* positions.
        features = [f1, f2, f3, f4, f5, f6] + [0.0] * 24

        # call API
        result, status = call_api(features, model)

        if status == 200:
            pred = result.get("prediction")
            prob = result.get("fraud_probability")

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("🔍 Prediction Result")

            if pred == 1:
                st.error("⚠️ FRAUD DETECTED!")
            else:
                st.success("✅ LEGITIMATE TRANSACTION")

            st.markdown(f"<p class='probability-box'>Fraud Probability: {prob}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error(f"Request failed: {result.get('error') or result}")


# ===================================================
# CSV UPLOAD MODE
# ===================================================
if mode == "Upload CSV File (FAST MODE)":
    st.subheader("📂 Upload CSV File (FAST MODE)")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        # store true labels (if present) BEFORE dropping
        st.session_state.y_true = df["Class"].copy() if "Class" in df.columns else None

        # show original columns
        st.write("### Columns detected:")
        st.write(df.columns.tolist())

        # prepare features for display and for sending
        feat_df = prepare_features_df(df)

        st.write("### Preview (prepared features sent to model):")
        st.dataframe(feat_df.head())

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
# VISUALIZATIONS
# ===================================================
st.subheader("📊 Model Performance Visualizations")

# make sure y_true and out_df exist and align
if st.session_state.y_true is not None and st.session_state.out_df is not None:
    try:
        y = st.session_state.y_true.reset_index(drop=True).iloc[:len(st.session_state.out_df)]
        probs = st.session_state.out_df["fraud_probability"].astype(float)
        preds = st.session_state.out_df["prediction"].astype(int)

        # quick diagnostics
        st.write("### Diagnostics")
        st.write("Unique predictions:", preds.unique().tolist())
        st.write("Unique probabilities (sample):", list(pd.Series(probs).dropna().unique()[:10]))

        if pd.Series(probs).dropna().nunique() < 2:
            st.warning("Not enough variation in predicted probabilities to generate ROC/PR curves. Check model or feature ordering.")
        else:
            plot_roc_curve(y, probs)
            plot_precision_recall(y, probs)

        # confusion matrix needs discrete preds
        if preds.nunique() < 2:
            st.warning("Predictions are all the same class — confusion matrix will be uninformative.")
        else:
            plot_confusion_matrix(y, preds)

    except Exception as e:
        st.error(f"Visualization error: {e}")
else:
    st.info("📌 Visualizations appear only after a batch prediction **and** only when CSV contains the 'Class' label column or you previously uploaded a file with labels.")


# ===================================================
# END
# ===================================================
