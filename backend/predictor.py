import joblib
import numpy as np

class ModelLoader:
    def __init__(self, model_path="models/logreg.pkl"):
        self.model = joblib.load(model_path)

    def predict(self, features: list):
        arr = np.array(features).reshape(1, -1)
        pred = self.model.predict(arr)[0]
        try:
            prob = self.model.predict_proba(arr)[0][1]
        except:
            prob = 0.0
        return {
            "prediction": int(pred),
            "fraud_probability": float(prob)
        }
