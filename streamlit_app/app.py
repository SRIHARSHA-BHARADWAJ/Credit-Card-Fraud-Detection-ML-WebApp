# UPDATED FRONTEND (NO PLOTTING)

import streamlit as st
import requests
import numpy as np
import pandas as pd
import time
import math

# ===================================================
# PAGE CONFIG
# ===================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

# ===================================================
# SESSION STATE
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
# UI STYLING
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

# ===================================================
# SIDEBAR
# ===================================================
st.sidebar.header("⚙️ Choose Model")

try:
    models = requests.get(API_MODELS, timeout=5).json().get("available_models", ["logreg", "rf"])
except:
    models = ["logreg", "rf"]

model = st.sidebar.radio("Select a Machine Learning Model:", models)
mode = st.sidebar.selectbox(
    "Choose Input Method:",
    ["Manual Input (6 values)", "Upload CSV File (FAST MODE)"]
)

# ===================================================
# FEATURE ORDER
# ===================================================
KAGGLE_ORDER = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]


def prepare_features_df(df):
    df_local = df.copy().reset_index(drop=True)
    if "Class" in df_local.columns:
        df_local = df_local.drop(columns=["Class"])

    numeric = df_local.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric) >= 30:
        feat_df = df_local[numeric[:30]].copy()
        feat_df.columns = KAGGLE_ORDER
        return feat_df

    for i in range(30 - len(df_local.columns)):
        df_local[f"pad_{i}"] = 0.0

    feat_df = df_local.iloc[:, :30]
    feat_df.columns = KAGGLE_ORDER
    return feat_df


# ===================================================
# API CALLS
# ===================================================
def call_api(features, model_selected):
    try:
        r = requests.post(
            f"{API_SINGLE}?model={model_selected}",
            json={"features": features},
            timeout=10
        )
        return r.json(), r.status_code
    except Exception as e:
        return {"error": str(e)}, 500


def predict_in_chunks(df, model_name="rf", chunk_size=4000):
    feat_df = prepare_features_df(df)
    n = len(feat_df)
    chunks = math.ceil(n / chunk_size)

    preds, probs = [], []

    st.info(f"Processing {n:,} rows ...")
    progress = st.progress(0)

    for i in range(chunks):
        s, e = i * chunk_size, min((i + 1) * chunk_size, n)
        batch = feat_df.iloc[s:e].values.tolist()

        try:
            r = requests.post(
                f"{API_BATCH}?model={model_name}",
                json={"features": batch},
                timeout=300
            )
            out = r.json()
        except Exception as err:
            st.error(f"Error at chunk {i+1}: {err}")
            return None

        preds.extend(out.get("predictions", []))
        probs.extend(out.get("probabilities", []))

        progress.progress(e / n)

    out_df = df.reset_index(drop=True).copy().iloc[:n]
    out_df["prediction"] = preds
    out_df["fraud_probability"] = probs

    st.session_state.out_df = out_df
    return out_df


# ===================================================
# MANUAL MODE
# ===================================================
if mode == "Manual Input (6 values)":
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
            st.write("### Prediction:")
            st.write(result)
        else:
            st.error(result)


# ===================================================
# CSV MODE
# ===================================================
if mode == "Upload CSV File (FAST MODE)":
    st.subheader("📂 Upload CSV File (FAST MODE)")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview:")
        st.dataframe(df.head())

        if "Class" in df:
            st.session_state.y_true = df["Class"].copy()

        if st.button("🚀 Predict All Rows"):
            out_df = predict_in_chunks(df, model_name=model)

            if out_df is not None:
                st.success("Prediction complete!")
                st.dataframe(out_df.head())

                csv = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download CSV", csv, "predictions.csv")
