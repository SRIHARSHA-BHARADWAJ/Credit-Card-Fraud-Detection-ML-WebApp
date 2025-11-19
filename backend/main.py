from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for predicting fraud using Logistic Regression, Random Forest, SVM, KNN, Decision Tree",
    version="1.0.0"
)

# Path to models folder
MODEL_DIR = Path("models")

# Available models
AVAILABLE_MODELS = {
    "logreg": MODEL_DIR / "logreg.pkl",
    "rf": MODEL_DIR / "rf.pkl",
    "knn": MODEL_DIR / "knn.pkl",
    "dt": MODEL_DIR / "dt.pkl",
}

# Cache to store loaded models
MODEL_CACHE = {}

# Input format for predict endpoint
class FeatureInput(BaseModel):
    features: List[float]


# ----------------------------
# Load model dynamically
# ----------------------------
def load_model(model_name: str):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found.")

    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    model_path = AVAILABLE_MODELS[model_name]
    if not model_path.exists():
        raise HTTPException(status_code=400, detail=f"Model file missing: {model_path}")

    model = joblib.load(model_path)
    MODEL_CACHE[model_name] = model
    return model


# ----------------------------
# Home Route
# ----------------------------
@app.get("/")
def home():
    return {"message": "Fraud Detection API running successfully!"}


# ----------------------------
# Return available models
# ----------------------------
@app.get("/get-models")
def get_models():
    return {"available_models": list(AVAILABLE_MODELS.keys())}


# ----------------------------
# Predict Endpoint
# ----------------------------
@app.post("/predict")
def predict(input_data: FeatureInput, model: str = "logreg"):

    # Load model
    try:
        model_obj = load_model(model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Convert input to array
    x = np.array(input_data.features).reshape(1, -1)

    # Prediction
    pred = model_obj.predict(x)[0]

    # Probability
    try:
        prob = model_obj.predict_proba(x)[0][1]
        prob = float(prob)
    except:
        prob = "N/A"

    return {
        "model_used": model,
        "prediction": int(pred),
        "fraud_probability": prob
    }
