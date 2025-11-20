<h1 align="center" style="font-size:56px; font-weight:900; margin:0; padding:0; background: linear-gradient(90deg,#00E5FF,#00C6FF,#0072FF,#4A00E0); -webkit-background-clip:text; color:transparent; animation: slideIn 1.5s ease forwards; opacity:0;">
  CREDIT CARD FRAUD DETECTION — MACHINE LEARNING SYSTEM
</h1>

<style>
@keyframes slideIn {
  0% { transform: translateX(-150px); opacity: 0; }
  60% { transform: translateX(10px); opacity: 0.96; }
  100% { transform: translateX(0); opacity: 1; }
}
.subtitle-anim { animation: fadeUp 1.2s ease forwards; opacity:0; animation-delay:0.35s; }
@keyframes fadeUp {
  0% { transform: translateY(22px); opacity:0; }
  100% { transform: translateY(0); opacity:1; }
}
@media (max-width:800px){ h1{font-size:40px} .subtitle-anim{font-size:15px} }
</style>

<div align="center" class="subtitle-anim" style="font-size:18px; color:#6b7280; max-width:820px; margin:auto; margin-top:12px;">
FastAPI backend with Logistic Regression & Random Forest models, strict 30-feature inference (V1–V28, Amount, Time), and a premium Streamlit UI featuring an animated SVG gauge, chunked batch processing, and robust error-tolerant prediction pipeline.
</div>

<br>

<div align="center">

![Static Badge](https://img.shields.io/badge/ML-Fraud%20Detection-blue)
![Static Badge](https://img.shields.io/badge/Backend-FastAPI-009485)
![Static Badge](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)
![Static Badge](https://img.shields.io/badge/Mode-Real%20Time%20%2B%20Batch-purple)
![Static Badge](https://img.shields.io/badge/Status-Production%20Grade-brightgreen)

</div>

---

## 🌐 🚀 LIVE DEMO

<div align="center" style="margin-top:10px; margin-bottom:25px; display:flex; gap:20px; justify-content:center; flex-wrap:wrap;">

<a href="https://credit-card-fraud-detection-ml-webapp.onrender.com/docs" target="_blank">
  <img src="https://img.shields.io/badge/🔥%20FASTAPI%20DOCS-111111?style=for-the-badge&logo=fastapi&logoColor=%2300FFAA&labelColor=000000"/>
</a>

<a href="https://credit-card-fraud-detection-ml-webapp-gnsnbwocoytsclrh4hryvj.streamlit.app/" target="_blank">
  <img src="https://img.shields.io/badge/⚡%20STREAMLIT%20APP-0A84FF?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=001122"/>
</a>

<a href="https://credit-card-fraud-detection-ml-webapp.onrender.com" target="_blank">
  <img src="https://img.shields.io/badge/🚀%20OPEN%20LIVE%20APP-00FFC6?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=0f1724"/>
</a>

</div>

---

## 🧠 Project Overview
A production-ready **Credit Card Fraud Detection System**, built with:

- Logistic Regression & Random Forest ML models  
- FastAPI backend with real-time & batch endpoints  
- Premium Streamlit dark UI  
- Automatic feature alignment (V1–V28, Amount, Time)  
- Animated SVG probability gauge  
- Chunked 4k-row batch processing  
- Backend fallback handling (405/500 tolerant)  
- Model caching for faster inference  

### ✔ Capabilities
- Single prediction with probability + risk classification  
- Batch CSV inference for thousands of records  
- Downloadable results  
- Actionable recommendations (Allow / MFA / Block)  

---

## 🎯 Key Features

### 🔍 Real-Time Fraud Prediction
- Enter **6 features**, remaining **24 auto-padded**  
- Predict fraud → get:
  - Fraud probability  
  - Risk level (Low / Elevated / High)  
  - Recommended action  

---

### 📊 Bulk CSV Processing
- Upload Kaggle-style or raw numeric CSV  
- Auto-align to expected model order  
- Chunking (4,000 rows per batch)  
- Zero-crash resume-friendly design  
- Download predictions  

---

### 🌀 Animated Risk Gauge
- Cyan-to-blue gradient arc  
- GPU-smooth CSS transitions  
- Professional, compact center display  

---

### 🛡️ FastAPI Backend
- `/predict` → single inference  
- `/predict-batch` → multi-row inference  
- Robust Pydantic validation  
- Auto-download models from GitHub Releases  
- Handles missing/incorrect features gracefully  

---

## 🏗 Architecture Diagram

```
                 +----------------------------+
                 |     GitHub Repository      |
                 +-------------+--------------+
                               |
                               |  Push (main)
                               v
                 +----------------------------+
                 |        Render Cloud        |
                 |  FastAPI Backend Service   |
                 +-------------+--------------+
                               |
                               |  JSON Requests
                               v
                 +----------------------------+
                 |     ML Models (RF / LR)    |
                 +-------------+--------------+
                               |
                               | Probabilities / Predictions
                               v
                 +----------------------------+
                 |     Streamlit Frontend     |
                 | Real-Time + CSV prediction |
                 +----------------------------+
```

---

## 📂 Project Structure

```
Credit-Card-Fraud-Detection-ML-WebApp/
├── README.md
├── LICENSE
│
├── streamlit_app/
│   └── app.py                    # Premium UI frontend
│
├── backend/
│   ├── main.py                   # FastAPI backend
│   ├── models/                   # ML model files
│   └── requirements.txt
│
└── utils/
    └── utils_plots.py            # (Legacy)
```

---

## 📦 Dataset (Kaggle)
Dataset used: **Credit Card Fraud Detection – PCA-transformed dataset**  
Rows: **284,807**  
Fraud cases: **492 (0.17%)**

| Feature | Meaning |
|--------|---------|
| V1–V28 | PCA-transformed anonymized features |
| Amount | Transaction amount |
| Time | Transaction index |
| Class | 1 = Fraud, 0 = Legit |

---

## 🚀 Run Locally

### 🔹 Frontend (Streamlit)
```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

### 🔹 Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 🔹 API Endpoints
- POST `/predict?model=rf`
- POST `/predict-batch?model=rf`
- GET `/get-models`

---

## 🛠 Future Enhancements
- Full Docker containerization (frontend + backend)  
- CI/CD automation (GitHub Actions → Render Deploy)  
- JWT authentication  
- SHAP explainability graphs  
- Monitoring (Grafana + Prometheus)  
- GPU inference engine  
- Redis caching layer  

---

## 📝 License
MIT License © 2025 **SRIHARSHA-BHARADWAJ**

---

## 👨‍💻 Author
**Sriharsha Bharadwaj**  
B.E. — Artificial Intelligence & Machine Learning  
B.M.S. College of Engineering, Bengaluru  
📧 Email: **sriharsha.ai22@bmsce.ac.in**  
🔗 GitHub: **https://github.com/SRIHARSHA-BHARADWAJ**
