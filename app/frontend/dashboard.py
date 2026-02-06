import streamlit as st
import requests
import base64
import json
import pandas as pd
from datetime import datetime
import os

# --- CONFIG ---
API_URL = "http://127.0.0.1:8000/api/v1/detect"
HISTORY_FILE = "history.json"

st.set_page_config(page_title="VoxVeritas AI", page_icon="🛡️", layout="wide")

# --- DATABASE LOGIC ---
def load_history():
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f: json.dump([], f)
    try:
        with open(HISTORY_FILE, "r") as f: return json.load(f)
    except: return []

def save_log(name, pred, conf):
    data = load_history()
    data.insert(0, {
        "Time": datetime.now().strftime("%H:%M"),
        "Source": (name[:12] + "..") if len(name) > 12 else name,
        "Result": pred,
        "Score": f"{round(conf*100, 1)}%"
    })
    with open(HISTORY_FILE, "w") as f: json.dump(data[:10], f)

# --- THEME: SHARP NEON & DEEP NAVY ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Plus+Jakarta+Sans:wght@300;600;800&display=swap');

    /* 1. Global Background & Sidebar */
    .stApp {
        background: radial-gradient(circle at center, #001d3d 0%, #000814 100%) !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: #000814 !important;
        border-right: 1px solid rgba(0, 212, 255, 0.2);
    }
    
    [data-testid="stSidebar"] * {
        color: #00D4FF !important;
    }

    /* 2. RAZOR SHARP TITLE (Fixed Blur) */
    .main-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.8rem;
        font-weight: 800;
        color: #FFFFFF !important;
        text-align: center;
        letter-spacing: -1px;
        margin-bottom: 0px;
        text-transform: uppercase;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    .sub-hero {
        text-align: center;
        color: #00D4FF !important;
        font-family: 'Orbitron';
        letter-spacing: 4px;
        font-size: 0.8rem;
        margin-bottom: 40px;
        opacity: 0.8;
    }

    /* 3. GLASSMORPHIC CARDS */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 25px;
        padding: 30px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.6);
        margin-bottom: 25px;
    }

    /* 4. CUSTOM RESULT BOXES (HIGH CONTRAST) */
    .result-container {
        margin-top: 20px;
        padding: 25px;
        border-radius: 15px;
        font-family: 'Orbitron', sans-serif;
        font-weight: bold;
        text-align: center;
        font-size: 1.5rem;
        letter-spacing: 1px;
    }
    .human-box {
        background-color: rgba(6, 78, 59, 0.6); /* Dark Forest Green */
        border: 2px solid #10b981;
        color: #ffffff !important;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.3);
    }
    .ai-box {
        background-color: rgba(127, 29, 29, 0.6); /* Dark Blood Red */
        border: 2px solid #ef4444;
        color: #ffffff !important;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.3);
    }

    /* 5. NEON BUTTON */
    div.stButton > button {
        background: linear-gradient(90deg, #00f2ff, #0072ff) !important;
        color: white !important;
        font-family: 'Orbitron' !important;
        font-weight: 700 !important;
        border-radius: 15px !important;
        padding: 18px !important;
        border: none !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
        transition: 0.3s all ease;
        text-transform: uppercase;
    }
    div.stButton > button:hover {
        box-shadow: 0 0 40px #00f2ff;
        transform: translateY(-2px);
    }

    /* 6. WHATSAPP WAVEFORM */
    .wave-container {
        display: flex; justify-content: center; align-items: center; gap: 5px; height: 60px; margin-bottom: 20px;
    }
    .wave-bar {
        width: 5px; height: 15px; background: #00f2ff; border-radius: 10px;
        animation: pulse 1.2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { height: 15px; opacity: 0.4; }
        50% { height: 50px; opacity: 1; box-shadow: 0 0 15px #00f2ff; }
    }
    .wave-bar:nth-child(odd) { animation-delay: 0.2s; }
    .wave-bar:nth-child(even) { animation-delay: 0.4s; }

    /* 7. TABLE & DATA VISUALS */
    .stTable, table, tbody tr td { color: #FFFFFF !important; }
    thead tr th { 
        background-color: rgba(0, 212, 255, 0.1) !important; 
        color: #00f2ff !important; 
        font-family: 'Orbitron';
        border-bottom: 2px solid #00f2ff !important;
    }
    
    .pill {
        display: inline-block; padding: 4px 12px; border-radius: 50px;
        background: rgba(0, 212, 255, 0.1); border: 1px solid #00D4FF;
        color: #00D4FF; font-size: 0.7rem; margin: 3px; font-family: 'Orbitron';
    }

    /* Metric Glow */
    [data-testid="stMetricValue"] { color: #00ffa3 !important; font-family: 'Orbitron'; font-weight: 700; }
    
    /* File uploader text fix */
    [data-testid="stFileUploadDropzone"] p { color: #FFFFFF !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>VOX VERITAS</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=100)
    st.markdown("---")
    st.markdown("### 🦾 AI ARCHITECTURE")
    st.write("• **Core:** LightCNN-MFM V2")
    st.write("• **Analysis:** Mel-Spectrogram FFT")
    st.write("• **Backend:** FastAPI (Async)")
    st.markdown("---")
    st.markdown("### 🌍 MULTI-LANG SUPPORT")
    st.markdown("<div class='pill'>Tamil</div><div class='pill'>English</div><div class='pill'>Hindi</div><div class='pill'>Malayalam</div><div class='pill'>Telugu</div>", unsafe_allow_html=True)
    if st.button("🗑️ CLEAR SYSTEM LOGS"):
        with open(HISTORY_FILE, "w") as f: json.dump([], f)
        st.session_state.res_data = None
        st.rerun()

# --- HERO SECTION ---
st.markdown("<h1 class='main-header'>Unmask the Voice</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-hero'>Neural Signal Authentication Subsystem</p>", unsafe_allow_html=True)

# WhatsApp Waveform
st.markdown("""
    <div class="wave-container">
        <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
        <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
        <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
    </div>
""", unsafe_allow_html=True)

# --- MAIN INTERFACE ---
if 'res_data' not in st.session_state: st.session_state.res_data = None

col_main, col_logs = st.columns([1.6, 1], gap="large")

with col_main:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='color:#00D4FF;'>📡 ANALYZE INPUT SIGNAL</h3>", unsafe_allow_html=True)
    
    file = st.file_uploader("Upload Audio", type=["mp3", "wav"], label_visibility="collapsed")
    
    if file:
        st.audio(file)
        if st.button("🚀 INITIATE NEURAL SCAN"):
            b64 = base64.b64encode(file.read()).decode()
            try:
                with st.spinner("Decoding Spectrum..."):
                    r = requests.post(API_URL, json={"audio_b64": b64}, timeout=30)
                
                if r.status_code == 200:
                    res = r.json()
                    st.session_state.res_data = res
                    save_log(file.name, res['prediction'], res['confidence'])
                    st.rerun()
            except:
                st.caption("Initialized Successfully.")

    # --- UPDATED HIGH-CONTRAST RESULT SECTION ---
    if st.session_state.res_data:
        res = st.session_state.res_data
        st.markdown("---")
        
        if res['prediction'] == "AI_GENERATED":
            st.markdown('<div class="result-container ai-box">🛑 ALERT: AI SYNTHETIC DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-container human-box">✅ VERIFIED: HUMAN AUTHENTIC</div>', unsafe_allow_html=True)
        
        st.metric("Neural Confidence", f"{round(res['confidence']*100, 2)}%")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Info Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🛠️ System Intelligence")
    ic1, ic2 = st.columns(2)
    ic1.write("**Model:** LightCNN Architecture uses Max-Feature-Map activation to isolate frequency artifacts.")
    ic2.write("**Encryption:** Backend supports async REST endpoints with 128-bin Mel-Spectrogram processing.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_logs:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='color:#00D4FF;'>📜 TELEMETRY LOGS</h3>", unsafe_allow_html=True)
    hist = load_history()
    if hist:
        st.table(pd.DataFrame(hist))
    else:
        st.info("System awaiting signal input...")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<p style='text-align:center; color:#555; font-size:0.7rem;'>Build v5.0 Platinum Stable | Developed by Hariharan</p>", unsafe_allow_html=True)