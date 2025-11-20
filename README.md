<div align="center" style="padding:20px 0 10px 0;">
  <img src="https://readme-typing-svg.demolab.com?font=Montserrat&weight=900&size= Thirty&duration=1300&pause=300&color=00E5FF&center=true&vCenter=true&width=1100&lines=CREDIT+CARD+FRAUD+DETECTION;FASTAPI+%7C+STREAMLIT+%7C+MACHINE+LEARNING+SYSTEM;REAL-TIME+AND+BATCH+PREDICTION" />
</div>

<div align="center" style="height:4px; width:330px; background:linear-gradient(90deg,#00E5FF,#0078FF,#00E5FF); border-radius:6px; filter:drop-shadow(0 0 10px #00E5FF); margin-top:6px;"></div>

<div align="center" style="font-size:15.5px; color:#8fa3b5; margin-top:14px; margin-bottom:22px; max-width:850px; line-height:1.45;">
A clean, production-aligned fraud detection stack powered by FastAPI, Streamlit, and optimized ML inference using a strict 30-feature transaction signature.
</div>

<div align="center" style="margin-top:18px; display:flex; gap:18px; justify-content:center; flex-wrap:wrap;">

<a href="https://credit-card-fraud-detection-ml-webapp.onrender.com/docs" target="_blank" style="text-decoration:none;">
  <img src="https://img.shields.io/badge/FASTAPI%20DOCS-000000?style=for-the-badge&logo=fastapi&logoColor=00FFAA" />
</a>

<a href="https://credit-card-fraud-detection-ml-webapp-gnsnbwocoytsclrh4hryvj.streamlit.app" target="_blank" style="text-decoration:none;">
  <img src="https://img.shields.io/badge/STREAMLIT%20LIVE-001122?style=for-the-badge&logo=streamlit&logoColor=FF4B4B" />
</a>

</div>

<br />

<p align="center">
  <img src="https://img.shields.io/badge/ML-Fraud%20Detection-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-009485?style=flat-square" />
  <img src="https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=flat-square" />
  <img src="https://img.shields.io/badge/Mode-RealTime%20%2B%20Batch-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Production%20Grade-brightgreen?style=flat-square" />
</p>

---

## 🧠 Project Overview — Technical Summary

A fully engineered, production-aligned **Credit Card Fraud Detection System** leveraging optimized ML inference and robust API workflows. **Credit Card Fraud Detection System** built using:

* FastAPI backend (Logistic Regression + Random Forest)
* Streamlit premium UI with SVG gauge
* Strict 30‑feature inference alignment (V1–V28, Amount, Time)
* Error‑tolerant prediction pipeline
* High‑throughput batch inference (4K rows per chunk)

---

## ✔ System Capabilities

* Real‑time single transaction scoring
* Fraud probability % + risk classification
* CSV batch processing (any size)
* Automatic feature ordering & padding
* Recommendations based on sensitivity threshold

---

## 🎯 Key Technical Features

### 🔍 Real‑Time Prediction

* Enter 6 features → remaining 24 padded automatically
* Output includes: prediction, probability, recommendation

### 📊 Batch CSV Processing

* Upload CSV with any column order
* Auto‑aligned to model order
* Chunk‑based processing (4000 rows)
* Downloadable predictions CSV

### 🌀 SVG Animated Gauge

* Compact + smooth stroke animation
* Gradient glow
* Perfect for dashboards

### 🛡️ FastAPI Backend

* Pydantic validation
* Model caching
* Graceful 405/500 fallback handling

---

## 🏗 System Architecture

```
Streamlit UI  →  FastAPI backend  →  ML models (RF / LR)  →  Probability + Prediction
```

---

## 📂 Repository Structure

```
Credit-Card-Fraud-Detection-ML-WebApp/
├── README.md
├── LICENSE
├── streamlit_app/
│   └── app.py
├── backend/
│   ├── main.py
│   ├── models/
│   └── requirements.txt
└── utils/
    └── utils_plots.py
```

---

## 📦 Dataset Specification

Kaggle Credit Card Fraud Dataset (284,807 rows):

* V1–V28: PCA features
* Amount
* Time
* Class (0 = Legit, 1 = Fraud)

---

## 🚀 Local Development Setup

### Frontend

```
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

### Backend

```
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

* POST /predict?model=rf
* POST /predict-batch?model=rf
* GET /get-models

---

## 🔮 Planned Enhancements

* Dockerization
* GitHub Actions CI/CD
* Authentication (API keys/JWT)
* SHAP explainability
* Redis caching
* GPU inference
* Monitoring dashboards

---

## 📝 License

MIT License © 2025 **SRIHARSHA‑BHARADWAJ**

---

## 👨‍💻 Maintainer Information

**Sriharsha Bharadwaj**
AI & ML — B.M.S. College of Engineering
📧 [sriharsha.ai22@bmsce.ac.in](mailto:sriharsha.ai22@bmsce.ac.in)
🔗 [https://github.com/SRIHARSHA-BHARADWAJ](https://github.com/SRIHARSHA-BHARADWAJ)
