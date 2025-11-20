<h1 align="center">
  <span style="background: linear-gradient(90deg,#00C6FF,#0072FF,#4A00E0,#8E2DE2); 
               -webkit-background-clip: text; 
               color: transparent; 
               font-size: 48px; 
               font-weight: 900; 
               display:inline-block;
               animation: slide 3s infinite alternate ease-in-out;">
  🔥 CREDIT CARD FRAUD DETECTION — Machine Learning • Streamlit • FastAPI
  </span>
</h1>

<style>
@keyframes slide {
  0% { transform: translateX(-6px); }
  100% { transform: translateX(6px); }
}
</style>

<div align="center">

![Static Badge](https://img.shields.io/badge/ML-Fraud%20Detection-blue)
![Static Badge](https://img.shields.io/badge/Backend-FastAPI-009485)
![Static Badge](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)
![Static Badge](https://img.shields.io/badge/Input-RealTime%20%2B%20Batch-purple)
![Static Badge](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

</div>

---

## 🌐 ⚡ **LIVE DEMO**

<div align="center">
  <a href="https://credit-card-fraud-detection-ml-webapp.onrender.com" target="_blank">
    <img src="https://img.shields.io/badge/🚀 LIVE%20APP-00FF9C?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=101010" 
         alt="Live Demo Button"
         style="animation: glow 2s infinite alternate;">
  </a>
</div>

<style>
@keyframes glow {
  0% { filter: drop-shadow(0px 0px 4px #00FF9C); }
  100% { filter: drop-shadow(0px 0px 12px #00FF9C); }
}
</style>

---

## 🧠 **Project Overview**

A production-ready **Credit Card Fraud Detection System** featuring:

- **FastAPI backend** for single & batch inference  
- **Streamlit premium dashboard** with dark, compact UI  
- **Machine Learning (Logistic Regression & Random Forest)**  
- **Fully automated feature ordering (V1–V28, Amount, Time)**  
- **Real-time probability scoring**  
- **Animated circular fraud gauge**  
- **CSV batch prediction support**  
- **Optimized chunk-based inference for large datasets**  
- **Backend fallback handling for 405/500 errors**

This system detects fraudulent transactions based on PCA-transformed financial features from the Kaggle Credit Card Fraud dataset.

---

## 🎯 **Key Features**

### 🔍 1. Real-Time Fraud Detection
Provide 6 inputs → system pads remaining 24 → backend returns:

- Fraud prediction (0/1)  
- Probability (%)  
- Risk level (Low / Elevated / High)  
- Actionable guidance  

---

### 📊 2. Bulk CSV Fraud Analysis
Upload a CSV → backend automatically:

- Aligns PCA feature order  
- Performs chunked predictions  
- Returns predictions + probabilities  
- Lets you download the final CSV  

Handles **4,000 rows per chunk** → supports **100k+ rows smoothly**.

---

### 🌀 3. High-Fidelity Animated Risk Gauge
A custom SVG gauge showing:

- Probability  
- Neon gradient  
- Center label  
- Smooth stroke animation  
- Glow highlights  

---

### 🛡️ 4. FastAPI Backend
- Model caching  
- Large JSON body handling  
- Pydantic validation  
- Error-safe fallback handling  
- Endpoint-based model selection  

---

### 🖥 5. Streamlit Dark Premium UI
- Clean liquid-glass panels  
- Perfect spacing  
- Professional AIML styling  
- Optional backend logs  
- Compact, responsive, modern  

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
           | Logistic Regression/RF  |
           +-----------+-------------+
                       |
                       | Prediction + Probability
                       v
           +-------------------------+
           |     Streamlit UI App    |
           | Real-time & CSV modes   |
           +-------------------------+


---

## 📂 **Project Structure**

Credit-Card-Fraud-Detection-ML-WebApp/
│
├── README.md # Documentation
├── LICENSE # MIT License
│
├── streamlit_app/
│ └── app.py # Streamlit Dashboard (Final Version)
│
├── backend/
│ ├── main.py # FastAPI backend
│ ├── models/ # ML Model Files (pkl)
│ └── requirements.txt # Backend dependencies
│
└── utils/
└── utils_plots.py # (Legacy)


---

## 🧪 **Dataset (Kaggle)**

| Feature | Description |
|--------|-------------|
| V1–V28 | PCA-transformed features |
| Amount | Transaction amount |
| Time   | Time index |
| Class  | 1 = Fraud, 0 = Legit |

---

## 🚀 **Run Locally — Frontend**

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py

---
## 🚀 **Run Locally — Backend**

cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
---

🧭 API Endpoints
▶ Single Prediction

POST /predict?model=rf

▶ Batch Prediction

POST /predict-batch?model=rf

▶ List Models

GET /get-models

---

🚀 Deployment

Deployed using Render Cloud

FastAPI backend → Web Service

Streamlit App → Web App

Models pulled from GitHub Releases

---

🛠 Future Enhancements

Docker containerization

GitHub Actions CI/CD

SHAP Explainability Dashboard

GPU-based vectorized inference

User authentication

Model A/B testing

Redis-based caching

Historical fraud analytics 

analytics

---

📝 License

MIT License © 2025 SRIHARSHA-BHARADWAJ

---

👨‍💻 Author

Sriharsha Bharadwaj
B.E. Artificial Intelligence & Machine Learning
B.M.S. College of Engineering, Bengaluru

📧 sriharsha.ai22@bmsce.ac.in

🔗 GitHub: https://github.com/SRIHARSHA-BHARADWAJ