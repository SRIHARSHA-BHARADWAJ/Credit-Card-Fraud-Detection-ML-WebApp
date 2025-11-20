# app.py
# FINAL FORM — Single-Center Modular Layout (Hybrid Cyan + Subtle Green Accent)
# - Centered animated gauge at top
# - Manual inputs & trigger below gauge
# - Collapsible CSV batch scoring block
# - Logs inside an expander (hidden by default)
# - Robust backend fallback (handles 405)
# - Feature-ordering/padding logic preserved
# - Compact, no empty glass blocks, ultra-premium dark styling

import streamlit as st
import requests
import numpy as np
import pandas as pd
import math
import time
from typing import List

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------
# Ultra-polished centered CSS (hybrid cyan + green)
# ----------------------------
st.markdown(
    """
    <style>
    :root{
      --bg: #06080a;
      --muted: #98a3ad;
      --accent-cyan: #00b7ff;
      --accent-green: #6aff88;
      --white-soft: rgba(245,248,250,0.96);
    }
    html, body, .stApp {
      background:
        radial-gradient(900px 420px at 8% 12%, rgba(4,8,12,0.36), transparent 8%),
        radial-gradient(700px 380px at 92% 88%, rgba(6,10,14,0.28), transparent 8%),
        var(--bg);
      color: var(--white-soft);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }

    /* center container */
    .center-wrap {
      display: flex;
      justify-content: center;
      margin-top: 18px;
      margin-bottom: 6px;
    }

    .center-panel {
      width: 980px;
      max-width: calc(100% - 40px);
      border-radius: 16px;
      padding: 18px 22px;
      background: linear-gradient(180deg, rgba(255,255,255,0.018), rgba(255,255,255,0.01));
      border: 1px solid rgba(255,255,255,0.035);
      box-shadow: 0 18px 48px rgba(0,0,0,0.6);
      backdrop-filter: blur(8px) saturate(120%);
    }

    /* header row */
    .hdr {
      display:flex; align-items:center; justify-content:space-between; gap:12px;
    }
    .hdr-title { font-size:28px; font-weight:800; margin:0; }
    .hdr-sub { color:var(--muted); font-size:13px; }

    /* main gauge area */
    .gauge-stage { display:flex; gap:24px; align-items:center; justify-content:center; padding:12px 6px; }
    .gauge-card { min-width:320px; display:flex; flex-direction:column; align-items:center; gap:6px; }

    /* input group */
    .inputs-row { display:flex; gap:12px; justify-content:center; margin-top:6px; flex-wrap:wrap; }
    .input-col { display:flex; flex-direction:column; gap:8px; min-width:220px; }

    /* compact controls */
    .run-btn > button { background: linear-gradient(90deg, rgba(0,183,255,0.10), rgba(106,255,136,0.06)); border-radius:10px; padding:8px 12px; border:1px solid rgba(0,183,255,0.06); font-weight:800; color:var(--white-soft); }
    .muted { color:var(--muted); font-size:13px; }

    /* badges */
    .badge { padding:6px 10px; border-radius:999px; font-weight:800; }
    .badge-low { background: rgba(106,255,136,0.04); color:#05a457; border:1px solid rgba(106,255,136,0.06); }
    .badge-med { background: rgba(0,183,255,0.03); color:#0aa6ff; border:1px solid rgba(0,183,255,0.05); }
    .badge-high { background: rgba(255,90,90,0.04); color:#ff6b6b; border:1px solid rgba(255,90,90,0.06); box-shadow: 0 10px 30px rgba(255,90,90,0.03); }

    /* small footer credit */
    .credit { text-align:center; color:var(--muted); font-size:13px; margin-top:12px; }
    .credit a { color: var(--accent-cyan); text-decoration:none; font-weight:700; }

    /* responsive */
    @media (max-width: 900px) {
      .center-panel { padding:14px; }
      .hdr-title { font-size:22px; }
      .gauge-card { min-width:260px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Feature ordering helper (unchanged)
# ----------------------------
KAGGLE_ORDER = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]


def prepare_features_df(df: pd.DataFrame) -> pd.DataFrame:
    df_local = df.copy().reset_index(drop=True)
    if "Class" in df_local.columns:
        df_local = df_local.drop(columns=["Class"])
    present = [c for c in KAGGLE_ORDER if c in df_local.columns]
    if len(present) >= 12:
        for c in KAGGLE_ORDER:
            if c not in df_local.columns:
                df_local[c] = 0.0
        return df_local[KAGGLE_ORDER].astype(float)
    numeric = df_local.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric) >= 30:
        feat = df_local[numeric[:30]].copy()
        feat.columns = KAGGLE_ORDER
        return feat
    for i in range(30 - df_local.shape[1]):
        df_local[f"pad_{i}"] = 0.0
    feat_df = df_local.iloc[:, :30].copy()
    feat_df.columns = KAGGLE_ORDER
    return feat_df.fillna(0.0).astype(float)


# ----------------------------
# Animated SVG gauge (refined)
# ----------------------------
def render_gauge(percent: float, size: int = 260) -> str:
    p = max(0.0, min(100.0, float(percent)))
    r = 92
    circ = 2 * math.pi * r
    filled = (p / 100.0) * circ
    empty = circ - filled
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center">
      <svg width="{size}" height="{size}" viewBox="0 0 240 240">
        <defs>
          <linearGradient id="gC" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#6aff88"/>
            <stop offset="55%" stop-color="#00b7ff"/>
            <stop offset="100%" stop-color="#ff9a66"/>
          </linearGradient>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="6" result="coloredBlur"/>
            <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>
        <g transform="translate(120,120)">
          <circle r="{r}" fill="none" stroke="rgba(255,255,255,0.03)" stroke-width="18"/>
          <circle r="{r}" fill="none" stroke="url(#gC)" stroke-width="18" stroke-linecap="round"
            stroke-dasharray="{filled} {empty}" stroke-dashoffset="{circ*0.25}" transform="rotate(-90)"
            style="transition: stroke-dasharray 900ms cubic-bezier(.2,.9,.3,1), stroke 900ms ease;" filter="url(#glow)"/>
          <circle r="66" fill="rgba(8,10,12,0.6)" stroke="rgba(255,255,255,0.02)" stroke-width="1.2"/>
          <text x="0" y="-10" text-anchor="middle" font-size="32" font-weight="800" fill="#e6f7ff">{p:.2f}%</text>
          <text x="0" y="22" text-anchor="middle" font-size="12" fill="#98a3ad">Fraud likelihood</text>
        </g>
      </svg>
    </div>
    """


# ----------------------------
# Backend endpoints & fallback helpers
# ----------------------------
API_BASE = "https://credit-card-fraud-detection-ml-webapp.onrender.com"
API_SINGLE = API_BASE + "/predict"
API_BATCH = API_BASE + "/predict-batch"
API_MODELS = API_BASE + "/get-models"


def try_get_models():
    try:
        r = requests.get(API_MODELS, timeout=4)
        r.raise_for_status()
        return r.json().get("available_models", ["logreg", "rf"])
    except Exception:
        return ["logreg", "rf"]


models_list = try_get_models()


def post_single_with_fallback(features: List[float], model_selected: str):
    payload = {"features": features}
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(f"{API_SINGLE}?model={model_selected}", json=payload, headers=headers, timeout=12)
    except Exception as e:
        return {"error": str(e)}, 500
    if r.status_code == 200:
        try:
            return r.json(), 200
        except:
            return {"detail": r.text}, r.status_code
    if r.status_code == 405:
        try:
            r2 = requests.post(API_SINGLE, json=payload, headers=headers, timeout=12)
        except Exception as e:
            return {"error": str(e)}, 500
        try:
            return r2.json(), r2.status_code
        except:
            return {"detail": r2.text}, r2.status_code
    try:
        return r.json(), r.status_code
    except:
        return {"detail": r.text}, r.status_code


def post_batch_with_fallback(batch: List[List[float]], model_selected: str, timeout: int = 300):
    payload = {"features": batch}
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(f"{API_BATCH}?model={model_selected}", json=payload, headers=headers, timeout=timeout)
    except Exception as e:
        return {"error": str(e)}, 500
    if r.status_code == 200:
        try:
            return r.json(), 200
        except:
            return {"detail": r.text}, r.status_code
    if r.status_code == 405:
        try:
            r2 = requests.post(API_BATCH, json=payload, headers=headers, timeout=timeout)
        except Exception as e:
            return {"error": str(e)}, 500
        try:
            return r2.json(), r2.status_code
        except:
            return {"detail": r2.text}, r2.status_code
    try:
        return r.json(), r.status_code
    except:
        return {"detail": r.text}, r.status_code


# ----------------------------
# Session state init
# ----------------------------
if "last_single" not in st.session_state:
    st.session_state.last_single = None
if "last_prob" not in st.session_state:
    st.session_state.last_prob = None
if "out_df" not in st.session_state:
    st.session_state.out_df = None
if "logs" not in st.session_state:
    st.session_state.logs = []

# ----------------------------
# Centered container layout
# ----------------------------
st.markdown('<div class="center-wrap"><div class="center-panel">', unsafe_allow_html=True)

# header
st.markdown('<div class="hdr"><div><h1 class="hdr-title">Credit Card Fraud Detection</h1></div></div>', unsafe_allow_html=True)

# top gauge stage
st.markdown('<div class="gauge-stage">', unsafe_allow_html=True)
# show gauge (initially empty if no result)
if st.session_state.get("last_prob") is not None:
    st.components.v1.html(render_gauge(st.session_state.last_prob, size=260), height=340)
else:
    # show placeholder gauge at 0.00% so layout is stable (not empty)
    st.components.v1.html(render_gauge(0.0, size=260), height=340)
st.markdown('</div>', unsafe_allow_html=True)

# inputs area (compact)
st.markdown('<div style="display:flex;justify-content:center;margin-top:8px;">', unsafe_allow_html=True)
st.markdown('<div style="max-width:880px;width:100%;">', unsafe_allow_html=True)

st.markdown('<div style="display:flex;justify-content:center;align-items:flex-start;gap:20px;flex-wrap:wrap;">', unsafe_allow_html=True)

# input columns
cols = st.columns([1, 1, 0.7])
with cols[0]:
    f1 = st.number_input("Feature 1", value=0.0, format="%.6f", key="f1")
    f2 = st.number_input("Feature 2", value=0.0, format="%.6f", key="f2")
with cols[1]:
    f3 = st.number_input("Feature 3", value=0.0, format="%.6f", key="f3")
    f4 = st.number_input("Feature 4", value=0.0, format="%.6f", key="f4")
with cols[2]:
    f5 = st.number_input("Feature 5", value=0.0, format="%.6f", key="f5")
    f6 = st.number_input("Feature 6", value=0.0, format="%.6f", key="f6")

st.markdown('</div>', unsafe_allow_html=True)

# run and sensitivity controls (compact)
controls = st.columns([1, 0.6, 0.6])
with controls[0]:
    run = st.button("Run prediction", key="run_single")
with controls[1]:
    sensitivity = st.slider("Sensitivity", 0.0, 100.0, 60.0, 1.0)
with controls[2]:
    show_raw = st.checkbox("Show raw", value=False)

st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

# handle run
if run:
    features = [f1, f2, f3, f4, f5, f6] + [0.0] * 24
    st.info("Requesting prediction...")
    out, stcode = post_single_with_fallback(features, model_selected=(st.session_state.get("model") if st.session_state.get("model") else "rf"))
    # note: we store model selection in sidebar below; fallback used here if not set
    if stcode != 200:
        st.error(f"Backend error (status {stcode}).")
        st.write(out)
        st.session_state.logs.append({"type": "error", "status": stcode, "detail": out})
        if stcode == 405:
            st.warning("405: Method Not Allowed. Ensure backend accepts POST /predict and CORS allows requests.")
            st.code(f"curl -X POST '{API_SINGLE}?model=rf' -H 'Content-Type: application/json' -d '{{\"features\":[0,0,...]}}'", language="bash")
    else:
        st.session_state.last_single = out
        prob = out.get("fraud_probability", None)
        try:
            prob_pct = round(float(prob) * 100.0, 2) if prob is not None else None
        except:
            prob_pct = None
        st.session_state.last_prob = prob_pct
        st.success("Result")
        if prob_pct is None:
            st.markdown('<div class="muted">Discrete prediction only (no probability).</div>', unsafe_allow_html=True)
        else:
            hp = sensitivity
            mp = sensitivity * 0.6
            if prob_pct >= hp:
                st.markdown(f"<div class='badge badge-high'>HIGH — {prob_pct}%</div>", unsafe_allow_html=True)
            elif prob_pct >= mp:
                st.markdown(f"<div class='badge badge-med'>ELEVATED — {prob_pct}%</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='badge badge-low'>LOW — {prob_pct}%</div>", unsafe_allow_html=True)
        if show_raw:
            st.json(out)

st.markdown('<div style="height:12px" />', unsafe_allow_html=True)

# CSV batch in collapsible expander
st.markdown('<div style="display:flex;justify-content:center;margin-top:6px;">', unsafe_allow_html=True)
with st.expander("Bulk scoring (CSV) — expand"):
    st.markdown('<div style="padding:6px 2px;">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV for batch scoring", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Cannot read CSV: {e}")
            df = None
        if df is not None:
            st.write("Preview:")
            st.dataframe(df.head())
            if "Class" in df.columns:
                st.session_state.y_true = df["Class"].copy()
            if st.button("Run batch scoring", key="batch_run"):
                st.info("Preparing batches...")
                feat_df = prepare_features_df(df)
                n = len(feat_df)
                st.write(f"Prepared {n} rows.")
                chunk_size = 4000
                preds, probs = [], []
                pbar = st.progress(0)
                status = st.empty()
                start = time.time()
                error = False
                for i in range(math.ceil(n / chunk_size)):
                    s = i * chunk_size
                    e = min((i + 1) * chunk_size, n)
                    batch = feat_df.iloc[s:e].values.tolist()
                    status.info(f"Processing chunk {i+1}: rows {s}-{e}")
                    out, stcode = post_batch_with_fallback(batch, model_selected=(st.session_state.get("model") if st.session_state.get("model") else "rf"), timeout=300)
                    if stcode != 200:
                        st.error(f"Batch failed at chunk {i+1}: status {stcode}")
                        st.write(out)
                        st.session_state.logs.append({"type": "error", "status": stcode, "detail": out})
                        error = True
                        break
                    preds.extend(out.get("predictions", []))
                    probs.extend(out.get("probabilities", []))
                    pbar.progress(min(1.0, e / n))
                    time.sleep(0.06)
                if not error:
                    out_df = df.reset_index(drop=True).iloc[:n].copy()
                    out_df["prediction"] = preds
                    out_df["fraud_probability"] = probs
                    st.session_state.out_df = out_df
                    st.success("Batch complete")
                    st.dataframe(out_df.head())
                    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download CSV", csv_bytes, "predictions.csv", "text/csv")
                    st.write(f"Processed {n} rows in {time.time()-start:.2f} s")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div></div></div>', unsafe_allow_html=True)  # close center panel & wrap

# ----------------------------
# Floating right-side compact controls (model selection + logs expander)
# ----------------------------
right_col = st.sidebar
right_col.markdown("### Settings")
model = right_col.radio("Model", try_get_models())
right_col.markdown("---")
right_col.markdown("### Logs")
with right_col.expander("Session logs (expand)"):
    if len(st.session_state.logs) == 0:
        right_col.markdown('<div class="muted">No logs yet.</div>', unsafe_allow_html=True)
    else:
        for entry in st.session_state.logs[-20:]:
            right_col.write(entry)

# footer credit (minimal)
st.markdown('<div class="credit">Developed by <a href="https://github.com/SRIHARSHA-BHARADWAJ" target="_blank">SRIHARSHA-BHARADWAJ</a></div>', unsafe_allow_html=True)
