# 🔥 CREDIT CARD FRAUD DETECTION — Machine Learning • Streamlit Dashboard • FastAPI Backend

<div align="center">

![Static Badge](https://img.shields.io/badge/ML-Fraud%20Detection-blue)
![Static Badge](https://img.shields.io/badge/Backend-FastAPI-009485)
![Static Badge](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)
![Static Badge](https://img.shields.io/badge/Mode-Real%20Time%20%2B%20Batch-purple)
![Static Badge](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

</div>

---

## 🌐 🚀 **LIVE WEB APPLICATION**

<div align="center">
  
### 👉 **[OPEN LIVE APP](https://credit-card-fraud-detection-ml-webapp.onrender.com)**

</div>

---

## 🧠 **Project Overview**

A production-ready **Credit Card Fraud Detection System** with:

- 🔹 Machine Learning Models (Logistic Regression, Random Forest)  
- 🔹 FastAPI Backend for single & batch inference  
- 🔹 Streamlit Premium Dark UI Dashboard  
- 🔹 Automatic feature ordering & padding  
- 🔹 Real-time probability scoring  
- 🔹 Animated circular risk gauge  
- 🔹 Bulk CSV prediction  
- 🔹 Clean architecture + strong validations  

This system detects **fraudulent transactions** by analyzing anonymized PCA-transformed banking features from the original Kaggle dataset.

### ✔ Core Capabilities
- **Single transaction prediction (instant)**
- **Batch CSV processing (thousands of rows)**
- **Normalized fraud probability (%)**
- **Risk-level classification (Low / Elevated / High)**
- **Adaptive sensitivity slider**
- **Backend fallback logic (handles 405/500 errors gracefully)**

---

## 🎯 **Key Features**

### 🔍 1. Real-Time Fraud Detection
Provide 6 feature inputs → system pads remaining 24 features → backend predicts:

- **Prediction (0 = Legit, 1 = Fraud)**
- **Probability (%)**
- **Risk level**
- **Guided recommendations**

---

### 📊 2. Bulk CSV Fraud Analysis
Upload a CSV with transaction records → system returns:

- Predictions  
- Fraud probabilities  
- Automatic alignment to V1–V28 + Amount + Time  
- Downloadable results file  

Handles **4,000 rows per chunk** via optimized FastAPI batching.

---

### 🌀 3. Animated Risk Gauge
A high-fidelity SVG circular gauge displays:

- Probability  
- Adaptive glow  
- Gradient stroke  
- Smooth animation  

---

### 🛡️ 4. Robust Backend (FastAPI)
- Handles large JSON payloads  
- Automatic model loading & caching  
- Logistic Regression + Random Forest available  
- Clean Pydantic models for validation  

---

### 🖥️ 5. Premium Streamlit UI
- Fully customized dark theme  
- Liquid-glass panels  
- Compact, centered layout  
- Responsive & minimal  
- Predictive guidance statements  
- Optional logs panel  

---

## 🏗️ **Architecture Diagram**

               +-------------------------+
               |    GitHub Repository    |
               +-----------+-------------+
                           |
                           |   Code Push
                           v
               +-------------------------+
               |         Render          |
               |   FastAPI Backend API   |
               +-----------+-------------+
                           |
                           |  JSON Request (POST)
                           v
               +-------------------------+
               |     ML Model (pkl)      |
               |  Logistic Regression /   |
               |     Random Forest        |
               +-----------+-------------+
                           |
                           |  Prediction + Probability
                           v
               +-------------------------+
               |     Streamlit UI App    |
               |  Real-time & CSV modes  |
               +-------------------------+


---

## 📂 **Project Structure**

Credit-Card-Fraud-Detection-ML-WebApp/
│
├── README.md # Documentation (You are here)
├── LICENSE # MIT License
│
├── streamlit_app/
│ └── app.py # Main Streamlit Dashboard (Final Premium UI)
│
├── backend/
│ ├── main.py # FastAPI app (prediction endpoints)
│ ├── models/ # Stored ML models (.pkl)
│ └── requirements.txt # Backend dependencies
│
└── utils/
└── utils_plots.py # (Optional older plotting utilities; not used anymore) 


---

## 🧪 **Dataset (Kaggle)**

Credit Card Fraud Dataset — 284,807 transactions

| Feature | Description |
|--------|-------------|
| `V1`–`V28` | PCA-transformed anonymized banking features |
| `Amount` | Transaction amount |
| `Time` | Time delta between transactions |
| `Class` | 0 = Legit, 1 = Fraud |

Dataset originally from Kaggle.

---

## 🚀 **Run Locally (Frontend)**

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py

## 🚀 **Run Locally (Backend)**
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000

🧭 API Endpoints
▶ Single Prediction

POST /predict?model=rf

▶ Batch Prediction

POST /predict-batch?model=rf

▶ Get Models

GET /get-models 

🚀 Deployment

Backend and frontend deployed on Render as:

FastAPI web service

Streamlit web app

Model files stored in GitHub Releases and auto-downloaded by backend.

🛠️ Future Enhancements

🔹 Add Docker support (Dockerfile + containerized backend)
🔹 Introduce GitHub Actions CI/CD
🔹 Add user authentication
🔹 Enable model versioning
🔹 Add Explainability (SHAP)
🔹 Add Fraud Score Calibration
🔹 Add vectorized GPU inference for ultra-high throughput
🔹 Add Redis caching for repeated predictions

📝 License

MIT License © 2025 SRIHARSHA-BHARADWAJ

👨‍💻 Author

Sriharsha Bharadwaj
B.E. — Artificial Intelligence & Machine Learning
B.M.S. College of Engineering, Bengaluru
📧mailto:sriharsha.ai22@bmsce.ac.in
🔗 GitHub: https://github.com/SRIHARSHA-BHARADWAJ