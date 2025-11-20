# 💳 Credit Card Fraud Detection — Machine Learning Web Application  
End-to-End Fraud Scoring System • Streamlit Frontend • FastAPI Backend • Real-Time Risk Analysis

---

# 🌐 Live Application  
🔗 **Frontend (Streamlit):** *Add your deployed Streamlit link*  
🔗 **Backend (FastAPI API):** *Add your Render backend link*

> The frontend communicates with the backend using `/predict` and `/predict-batch` endpoints to provide fast real-time and bulk fraud predictions.

---

# 🧠 Project Overview  
This project is a **Credit Card Fraud Detection System** built for real-world financial risk scoring.  
It uses machine learning models to classify whether a transaction is fraudulent based on **30 numerical features** derived from the popular Kaggle Credit Card Fraud Dataset.

The goal of the project is to provide:

- ✔ Real-time fraud detection  
- ✔ High-speed batch CSV analysis  
- ✔ Fraud probability scoring  
- ✔ Clean, stable, production-ready backend  
- ✔ Professional dark-themed frontend  
- ✔ Fully automated model loading + inference  

This system can be integrated into real fintech pipelines that require **fraud risk intelligence**, anomaly detection, or financial triage.

---

# 🎯 Key Features

## 🔍 1. Real-Time Single Prediction  
- Enter 6 numeric features  
- Remaining features auto-padded to the required 30  
- Model predicts **fraud probability (%)**  
- Risk classification:
  - **Low Risk**
  - **Elevated Risk**
  - **High Risk**
- Shows a compact fraud likelihood gauge  
- Gives final decision & recommendation

---

## 📂 2. Bulk CSV Prediction (Batch Mode)  
- Upload a CSV with:
  - Kaggle-style columns (V1–V28, Amount, Time) **OR**
  - Any numeric columns  
- System auto-aligns features to correct order  
- Processes thousands of rows in chunks  
- Generates fraud predictions + probabilities  
- Provides downloadable results CSV  

---

## ⚙️ 3. FastAPI Backend (Robust & Stable)  
- `/predict` → Single transaction scoring  
- `/predict-batch` → Bulk scoring  
- Auto-downloads models from GitHub Releases  
- Caches loaded models (faster inference)  
- Clean exception handling  
- Works flawlessly with Streamlit frontend  

---

## 🖥️ 4. Modern Streamlit UI  
- Clean professional dark theme  
- Fraud probability gauge  
- Easy manual input section  
- CSV upload & preview  
- Final judgement statements  
- Fully responsive layout  

---

# 🧱 Architecture Overview

User Input / CSV
↓
Streamlit Frontend (UI)
↓ JSON request (30 numerical features)
FastAPI Backend (Model Server)
↓
ML Model (Random Forest / Logistic Regression)
↓
Probability + Fraud/Legit Classification
↓
Frontend Visualization + Recommendations

yaml
Copy code

---

# 📂 Project Structure

Credit-Card-Fraud-Detection-ML-WebApp/
│
├── streamlit_app/
│ └── app.py # Main Streamlit UI
│
├── main.py # FastAPI backend
├── models/ # Cached downloaded ML models
├── requirements.txt # Python dependencies
└── README.md

yaml
Copy code

---

# 📦 Dataset (Kaggle Credit Card Fraud)  
The system uses the standard **30-feature PCA-transformed dataset**:

- **V1 – V28** — PCA components  
- **Amount**  
- **Time**  
- Target label `Class` (0 = legitimate, 1 = fraud)

The model expects **exactly 30 input values**.

The frontend ensures:
- CSV → aligned  
- Manual input → padded  

---

# 🚀 Local Development Setup

## 1️⃣ Clone Repository  
```bash
git clone https://github.com/SRIHARSHA-BHARADWAJ/Credit-Card-Fraud-Detection-ML-WebApp
cd Credit-Card-Fraud-Detection-ML-WebApp
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Run Backend (FastAPI)
bash
Copy code
uvicorn main:app --reload --port 10000
Backend will automatically:

Fetch models from GitHub releases

Cache them

Start inference server

4️⃣ Run Frontend (Streamlit)
bash
Copy code
streamlit run streamlit_app/app.py
Frontend will:

Connect to FastAPI backend

Provide UI for predictions

🐳 Docker & CI/CD (Future Enhancements)
Currently not implemented in this repository, but planned improvements include:

🚀 Docker Containerization
Dockerfile for backend

Dockerfile for frontend

Multi-stage builds

Optimized environment size

🔁 CI/CD Pipeline (GitHub Actions)
Auto-testing

Auto-linting

Auto-backend deployment

Auto Streamlit updates

🔮 Additional Enhancements
Model monitoring dashboard

Drift detection

Retraining pipeline

Fraud scoring reports

Kafka-based streaming ingestion

AWS Lambda deployment

gRPC model server

JWT-secured API endpoints

📡 API Documentation
▶️ POST /predict
Single sample scoring
Body:

json
Copy code
{
  "features": [30 float values]
}
Response:

json
Copy code
{
  "prediction": 0,
  "fraud_probability": 0.0134
}
▶️ POST /predict-batch
Batch scoring (CSV → list of lists)

json
Copy code
{
  "features": [
    [30 floats],
    [30 floats]
  ]
}
Response:

json
Copy code
{
  "predictions": [...],
  "probabilities": [...]
}
📝 License
This project is licensed under the MIT License.
You may use, modify, and distribute this software freely.

👨‍💻 Author
Sriharsha Bharadwaj
B.E. Artificial Intelligence & Machine Learning
B.M.S. College of Engineering, Bengaluru

📧 sriharsha.ai22@bmsce.ac.in
🔗 GitHub: https://github.com/SRIHARSHA-BHARADWAJ