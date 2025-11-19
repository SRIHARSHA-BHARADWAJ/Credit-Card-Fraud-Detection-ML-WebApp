import streamlit as st
import numpy as np
import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection System")
st.write("Upload a CSV file containing transaction data and choose a model to predict fraud.")

# ------------------------
# MODEL SELECTION
# ------------------------
model_choice = st.selectbox(
    "Select Model",
    ["logreg", "rf", "knn", "dt"]
)

# ------------------------
# CSV UPLOAD
# ------------------------
uploaded_file = st.file_uploader("Upload CSV (must contain 30 feature columns)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Uploaded Data Preview")
    st.dataframe(df)

    # Select row for prediction
    row_index = st.number_input(
        "Select row index for prediction",
        min_value=0,
        max_value=len(df)-1,
        value=0
    )

    selected_row = df.iloc[row_index].tolist()

    st.write("Selected Row Features (30 values):")
    st.write(selected_row)

    # ------------------------
    # PREDICT BUTTON
    # ------------------------
    if st.button("🔍 Predict Fraud for Selected Row"):
        try:
            payload = {"features": selected_row}

            response = requests.post(
                f"{API_URL}?model={model_choice}",
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                prediction = "FRAUD ❌" if result["prediction"] == 1 else "NOT FRAUD ✅"

                st.subheader("🔎 Prediction Result")
                st.success(f"Prediction: **{prediction}**")
                st.info(f"Fraud Probability: **{result['fraud_probability']}**")

            else:
                st.error("Backend Error: " + response.text)

        except Exception as e:
            st.error("API connection failed. Is backend running?")
else:
    st.warning("Upload a CSV file to continue.")
