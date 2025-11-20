# <p align="center"><img src="https://readme-typing-svg.demolab.com?font=Montserrat&weight=900&size=42&duration=2500&pause=800&color=00C6FF&center=true&vCenter=true&width=1000&lines=CREDIT+CARD+FRAUD+DETECTION;MACHINE+LEARNING+SYSTEM" /></p>

<div align="center" style="margin-top:18px; display:flex; gap:16px; justify-content:center; flex-wrap:wrap;">

<a href="https://credit-card-fraud-detection-ml-webapp.onrender.com/docs" target="_blank" style="text-decoration:none;">
  <img src="https://img.shields.io/badge/🔥%20FASTAPI%20DOCS-111111?style=for-the-badge&logo=fastapi&logoColor=%2300FFAA&labelColor=000000" />
</a>

<a href="https://credit-card-fraud-detection-ml-webapp-gnsnbwocoytsclrh4hryvj.streamlit.app/#model-performance-visualizations" target="_blank" style="text-decoration:none;">
  <img src="https://img.shields.io/badge/⚡%20STREAMLIT%20LIVE-0A84FF?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=001122" />
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

## 🧠 Project Overview

A fully engineered **Credit Card Fraud Detection System** built using:

* FastAPI backend (Logistic Regression + Random Forest)
* Streamlit premium UI with SVG gauge
* Strict 30-feature inference alignment (V1–V28, Amount, Time)
* Error-tolerant prediction pipeline
* High-throughput batch inference (4K rows per chunk)

---

## ✔ Capabilities

* Real-time single transaction scoring
* Fraud probability % + risk classification
* CSV batch processing (any size)
* Automatic feature ordering & padding
* Recommendations based on sensitivity threshold

---

## 🎯 Key Features

### 🔍 Real-Time Prediction

* Enter 6 features → remaining 24 padded automatically
* Output includes: prediction, probability, recommendation

### 📊 Batch CSV Processing

* Upload CSV with any column order
* Auto-aligned to model order
* Chunk-based processing (4000 rows)
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

## 🏗 Architecture

```
Streamlit UI  →  FastAPI backend  →  ML models (RF / LR)  →  Probability + Prediction
```

---

## 📂 Project Structure

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

## 📦 Dataset

Kaggle Credit Card Fraud Dataset (284,807 rows):

* V1–V28: PCA features
* Amount
* Time
* Class (0 = Legit, 1 = Fraud)

---

## 🚀 Run Locally

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

## 🔮 Future Enhancements

* Dockerization
* GitHub Actions CI/CD
* Authentication (API keys/JWT)
* SHAP explainability
* Redis caching
* GPU inference
* Monitoring dashboards

---

## 📝 License

MIT License © 2025 **SRIHARSHA-BHARADWAJ**

---

## 👨‍💻 Author

**Sriharsha Bharadwaj**
AI & ML — B.M.S. College of Engineering
📧 [sriharsha.ai22@bmsce.ac.in](mailto:sriharsha.ai22@bmsce.ac.in)
🔗 [https://github.com/SRIHARSHA-BHARADWAJ](https://github.com/SRIHARSHA-BHARADWAJ)
