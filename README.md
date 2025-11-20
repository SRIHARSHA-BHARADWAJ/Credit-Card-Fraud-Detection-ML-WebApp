<h1 align="center">
  <span style="
    background: linear-gradient(90deg,#0072FF,#00C6FF,#4A00E0,#8E2DE2);
    -webkit-background-clip:text;
    color:transparent;
    font-size:48px;
    font-weight:900;">
    CREDIT CARD FRAUD DETECTION — Machine Learning • FastAPI • Streamlit
  </span>
</h1>

<div align="center">

  
![ML](https://img.shields.io/badge/ML-Fraud%20Detection-blue)
![Backend](https://img.shields.io/badge/FastAPI-Backend-009485)
![Frontend](https://img.shields.io/badge/Streamlit-UI-FF4B4B)
![Mode](https://img.shields.io/badge/Mode-RealTime+Batch-purple)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)


<br><br>

<a href="https://credit-card-fraud-detection-ml-webapp.onrender.com">
  <img src="https://img.shields.io/badge/🚀 LIVE DEMO-000000?style=for-the-badge&logo=streamlit&logoColor=white" />
</a>

</div>

---

## 🧠 Project Overview

A production-ready **Credit Card Fraud Detection System** with:

- Machine Learning models (Logistic Regression, Random Forest)
- FastAPI backend supporting single + batch inference
- Streamlit premium UI (dark mode + compact + custom gauge)
- Real-time fraud probability scoring with risk classification
- Automatic 30-feature alignment (V1–V28 + Amount + Time)
- Robust backend fallback handling for 405/500 responses
- Batch CSV prediction with chunk processing (4000 rows/chunk)

This project uses the **Kaggle Credit Card Fraud dataset** with 284,807 transactions.

## 🎯 Key Features

### 🔍 1. Real-Time Prediction
- Enter 6 values → system pads to 30 features
- Predicts fraud probability (%)
- Shows Low, Elevated, or High risk
- Provides automated decision guidance

### 📊 2. Batch CSV Prediction
- Supports thousands of rows
- Auto-aligns Kaggle-style features
- Returns final predictions + fraud probability
- Allows result download as CSV

### 🌀 3. Animated Probability Gauge
- SVG circular gauge
- Gradient stroke + glow
- Smooth animation transitions

### 🧩 4. Backend (FastAPI)
- Pydantic validation
- Model caching (RF + LR)
- Handles large JSON safely

### 🖥️ 5. Streamlit UI
- Professional dark theme
- Liquid-glass panels
- Minimal + compact layout
- Optional backend logs
- Sensitivity threshold slider

## 🏗 Architecture Diagram

Streamlit UI (Frontend)
        |
        |   POST /predict or /predict-batch
        v
FastAPI Backend (Render)
        |
        |   Loads ML model (RF / LR)
        v
Model Inference Engine
        |
        v
Prediction + Probability Response

## 📂 Project Structure

Credit-Card-Fraud-Detection-ML-WebApp/
│
├── streamlit_app/
│   └── app.py                       
├── backend/
│   ├── main.py                    
│   ├── models/                     
│   └── requirements.txt
│
├── utils/
│   └── utils_plots.py            
│
├── LICENSE
└── README.md

## 🧪 Dataset (Kaggle)

| Feature | Description |
|--------|-------------|
| V1–V28 | PCA-transformed anonymized banking features |
| Amount | Transaction value |
| Time | Time delta since first transaction |
| Class | 0 = Legit, 1 = Fraud |

Dataset Source: Kaggle – Credit Card Fraud Detection.

## 🚀 Run Locally

### 🔹 Frontend (Streamlit)

```
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

### 🔹 Backend (FastAPI)

```
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 🔹 API Endpoints

* POST /predict?model=rf
* POST /predict-batch?model=rf
* GET /get-models

---

## 🛠 Future Enhancements

* Docker container for backend + frontend
* CI/CD with GitHub Actions
* User authentication & API keys
* Model versioning with experiment tracking
* SHAP explainability dashboard
* GPU-accelerated vectorized inference
* Redis caching layer for repeated queries
* Monitoring + logging dashboard (Prometheus/Grafana)

---

## 📝 License

MIT License © 2025 **SRIHARSHA-BHARADWAJ**

---

## 👨‍💻 Author

**Sriharsha Bharadwaj**
B.E. AIML — B.M.S. College of Engineering, Bengaluru
📧 [sriharsha.ai22@bmsce.ac.in](mailto:sriharsha.ai22@bmsce.ac.in)
🔗 GitHub: [https://github.com/SRIHARSHA-BHARADWAJ](https://github.com/SRIHARSHA-BHARADWAJ)
