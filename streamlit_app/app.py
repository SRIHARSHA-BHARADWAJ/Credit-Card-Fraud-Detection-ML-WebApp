import streamlit as st
import requests
import numpy as np
import pandas as pd
import time
import math
from typing import List

# --------------------------------------------
# PAGE CONFIG
# --------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection — Obsidian Quantum",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------
# DARK GLASS PREMIUM CSS (CLEANED + NO EMPTY BLOCKS)
# --------------------------------------------
st.markdown("""
<style>

:root{
  --bg:#0b0f14;
  --glass: rgba(255,255,255,0.03);
  --muted:#9aa5b1;
  --accent:#00c8ff;
  --accent2:#76ff03;
  --text:#e9f1f7;
}

/* Global background */
.stApp {
  background: radial-gradient(900px 600px at 20% 10%, rgba(0,180,255,0.18), transparent 60%),
              radial-gradient(700px 600px at 80% 90%, rgba(70,255,80,0.12), transparent 70%),
              var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial;
}

/* HERO (no empty padding, no empty blocks) */
.hero {
  background: var(--glass);
  padding: 22px 26px 18px 26px;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.04);
  backdrop-filter: blur(10px);
  box-shadow: 0 8px 30px rgba(0,0,0,0.45);
}
.title {
  font-size: 38px;         /* Hybrid size */
  font-weight: 900;
  letter-spacing: -0.5px;
  margin: 0;
}
.subtitle {
  font-size: 15px;
  color: var(--muted);
  margin-top: 6px;
  margin-bottom: 0;
}

/* CARDS (cleaned, no ghost blocks) */
.card {
  background: rgba(255,255,255,0.035);
  padding: 16px 18px;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.05);
  backdrop-filter: blur(8px);
  box-shadow: 0 6px 24px rgba(0,0,0,0.45);
  margin-bottom: 12px;
}

/* Buttons */
.stButton > button {
  background: linear-gradient(90deg, rgba(0,200,255,0.22), rgba(118,255,3,0.16));
  border: 1px solid rgba(0,200,255,0.16);
  border-radius: 10px;
  color: white;
  font-weight: 800;
  padding: 10px 14px;
}

/* Risk badges */
.badge-low {
  background: rgba(16,185,129,0.12);
  border: 1px solid rgba(16,185,129,0.18);
  color: #10b981;
  padding: 6px 12px;
  font-size: 16px;
  font-weight: 700;
  border-radius: 999px;
}
.badge-med {
  background: rgba(255,170,0,0.12);
  border: 1px solid rgba(255,170,0,0.18);
  color: #ffae00;
  padding: 6px 12px;
  font-size: 16px;
  font-weight: 700;
  border-radius: 999px;
}
.badge-high {
  background: rgba(239,68,68,0.12);
  border: 1px solid rgba(239,68,68,0.18);
  color: #ef4444;
  padding: 6px 12px;
  font-size: 17px;
  font-weight: 800;
  border-radius: 999px;
}

/* Muted */
.muted {
  font-size: 14px;
  color: var(--muted);
}

/* Credit footer */
.credit {
  text-align:center;
  margin-top:20px;
  color: var(--muted);
  font-size: 13px;
}
.credit a {
  color: var(--accent2);
  text-decoration:none;
  font-weight:800;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------
# GAUGE WIDGET (COMPACT VERSION)
# --------------------------------------------
def render_gauge(percent: float, size: int = 200):
    p = max(0, min(100, percent))
    r = 80
    circ = 2 * math.pi * r
    fill = circ * p / 100
    empty = circ - fill

    return f"""
    <div style="width:{size}px; margin:auto; text-align:center;">
      <svg width="{size}" height="{size}" viewBox="0 0 220 220">
        <defs>
          <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#00c8ff"/>
            <stop offset="100%" stop-color="#76ff03"/>
          </linearGradient>
        </defs>

        <g transform="translate(110,110)">
          <circle r="{r}" stroke="rgba(255,255,255,0.05)" stroke-width="18" fill="none"/>
          <circle r="{r}" stroke="url(#grad)" stroke-width="18" fill="none"
            stroke-dasharray="{fill} {empty}" stroke-linecap="round"
            transform="rotate(-90)"/>
          <text x="0" y="8" text-anchor="middle" font-size="30" fill="#eaf6ff" font-weight="700">{p:.2f}%</text>
        </g>
      </svg>
    </div>
    """

# --------------------------------------------
# API ENDPOINTS
# --------------------------------------------
API = "https://credit-card-fraud-detection-ml-webapp.onrender.com"
API_SINGLE = f"{API}/predict"
API_BATCH = f"{API}/predict-batch"
API_MODELS = f"{API}/get-models"

def get_models():
    try:
        r = requests.get(API_MODELS, timeout=4)
        return r.json().get("available_models", ["rf", "logreg"])
    except:
        return ["rf", "logreg"]

MODELS = get_models()

# --------------------------------------------
# FEATURE ORDER & PREP
# --------------------------------------------
ORDER = [f"V{i}" for i in range(1,29)] + ["Amount","Time"]

def prep_df(df):
    df = df.copy()
    if "Class" in df:
        df = df.drop("Class", axis=1)

    present = [c for c in ORDER if c in df.columns]
    if len(present) >= 12:
        for c in ORDER:
            if c not in df:
                df[c] = 0.0
        return df[ORDER].astype(float)

    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(nums) >= 30:
        out = df[nums[:30]].copy()
        out.columns = ORDER
        return out.astype(float)

    for i in range(30 - df.shape[1]):
        df[f"pad_{i}"] = 0.0

    out = df.iloc[:, :30]
    out.columns = ORDER
    return out.astype(float)

# --------------------------------------------
# POST REQUEST FUNCTIONS
# --------------------------------------------
def single_api(vec, mdl):
    try:
        r = requests.post(f"{API_SINGLE}?model={mdl}", json={"features":vec}, timeout=10)
        return r.json(), r.status_code
    except Exception as e:
        return {"error":str(e)}, 500

def batch_api(batch, mdl):
    try:
        r = requests.post(f"{API_BATCH}?model={mdl}", json={"features":batch}, timeout=300)
        return r.json(), r.status_code
    except Exception as e:
        return {"error":str(e)}, 500

# --------------------------------------------
# HERO (NO EMPTY BLOCKS)
# --------------------------------------------
st.markdown("<div class='hero'>", unsafe_allow_html=True)
st.markdown("<div class='title'>Credit Card Fraud Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Professional • Ultra-grade precision • Real-time fraud scoring</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------
# SIDEBAR
# --------------------------------------------
model = st.sidebar.radio("Model", MODELS)
mode = st.sidebar.selectbox("Mode", ["Single Prediction", "Bulk CSV Prediction"])
sensitivity = st.sidebar.slider("Risk Sensitivity %", 30, 90, 60)
show_logs = st.sidebar.checkbox("Show logs (optional)", False)

# --------------------------------------------
# SESSION
# --------------------------------------------
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last_prob" not in st.session_state:
    st.session_state.last_prob = None

# --------------------------------------------
# LAYOUT
# --------------------------------------------
left, right = st.columns([2,1])

# ###############################################################
# LEFT SIDE — SINGLE + BULK
# ###############################################################
with left:

    # --------------------------------------------
    # SINGLE PREDICTION BLOCK
    # --------------------------------------------
    if mode == "Single Prediction":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Single Prediction")
        st.markdown("<div class='muted'>Enter ANY 6 numeric values. Remaining features auto-padded.</div>", unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            f1 = st.number_input("Feature 1", 0.0)
            f2 = st.number_input("Feature 2", 0.0)
            f3 = st.number_input("Feature 3", 0.0)
        with c2:
            f4 = st.number_input("Feature 4", 0.0)
            f5 = st.number_input("Feature 5", 0.0)
            f6 = st.number_input("Feature 6", 0.0)

        if st.button("Run Prediction"):
            vec = [f1,f2,f3,f4,f5,f6] + [0]*24
            out,code = single_api(vec,model)

            if code != 200:
                st.error("Backend error. Try again.")
                st.session_state.logs.append({"error":out})
            else:
                prob = out["fraud_probability"]
                pct = round(prob*100,2)
                st.session_state.last_prob = pct

                # Risk classification
                high=sensitivity
                med=sensitivity*0.6

                if pct>=high:
                    st.markdown(f"<div class='badge-high'>HIGH RISK — {pct}%</div>", unsafe_allow_html=True)
                    st.markdown("<div class='muted'>Final judgement: Immediate block recommended.</div>", unsafe_allow_html=True)
                elif pct>=med:
                    st.markdown(f"<div class='badge-med'>ELEVATED — {pct}%</div>", unsafe_allow_html=True)
                    st.markdown("<div class='muted'>Final judgement: Secondary verification required.</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='badge-low'>LOW — {pct}%</div>", unsafe_allow_html=True)
                    st.markdown("<div class='muted'>Final judgement: Approved — low anomaly signature.</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------
    # BULK CSV
    # --------------------------------------------
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Bulk CSV Prediction")
        st.markdown("<div class='muted'>Upload a CSV. Features will be aligned and padded automatically.</div>", unsafe_allow_html=True)

        up = st.file_uploader("Upload", type=["csv"])
        if up:
            df = pd.read_csv(up)
            st.dataframe(df.head())

            if st.button("Run Bulk Prediction"):
                feat = prep_df(df)
                preds=[]
                probs=[]
                n=len(feat)
                pbar=st.progress(0)

                for i in range(math.ceil(n/4000)):
                    s=i*4000
                    e=min((i+1)*4000,n)
                    batch=feat.iloc[s:e].values.tolist()
                    out,code=batch_api(batch,model)
                    if code!=200:
                        st.error("Batch error")
                        break
                    preds+=out["predictions"]
                    probs+=out["probabilities"]
                    pbar.progress(e/n)

                df["prediction"]=preds
                df["fraud_probability"]=probs
                st.dataframe(df.head())

                # FINAL DATASET JUDGEMENT
                fraud_rate = sum(preds)/len(preds)

                if fraud_rate == 0:
                    st.success("Final dataset judgement: No fraud indicators — dataset clean.")
                elif fraud_rate < 0.10:
                    st.info("Final dataset judgement: Operating normal — minor anomalies only.")
                else:
                    st.error("Final dataset judgement: Dataset shows risky profile — review advised.")

                st.download_button(
                    "Download Results",
                    df.to_csv(index=False).encode("utf-8"),
                    "predictions.csv",
                    "text/csv"
                )

        st.markdown("</div>", unsafe_allow_html=True)


# ###############################################################
# RIGHT SIDE — GAUGE (+ optional logs)
# ###############################################################
with right:

    # Gauge only
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Gauge")
    if st.session_state.last_prob is not None:
        st.components.v1.html(render_gauge(st.session_state.last_prob,200),height=250)
    else:
        st.markdown("<div class='muted'>Run a single prediction to populate.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Optional logs
    if show_logs:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Logs")
        if len(st.session_state.logs)==0:
            st.markdown("<div class='muted'>No logs yet.</div>", unsafe_allow_html=True)
        else:
            for log in st.session_state.logs[-10:]:
                st.write(log)
        st.markdown("</div>", unsafe_allow_html=True)

# Footer credit
st.markdown("<div class='credit'>Developed by <a href='https://github.com/SRIHARSHA-BHARADWAJ'>SRIHARSHA-BHARADWAJ</a> · Obsidian Quantum Edition</div>", unsafe_allow_html=True)
