import streamlit as st
import librosa
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import io
from datetime import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TECHNOVA SOLUTION | Pro AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. SESSION STATE INITIALIZATION (ERROR FIX) ---
if 'active_pid' not in st.session_state:
    st.session_state.active_pid = None
if 'vibe_name' not in st.session_state:
    st.session_state.vibe_name = ""

# --- 3. LOAD ML MODELS (PDF LOGIC) ---
@st.cache_resource
def load_models():
    try:
        # Colab-la generate panna files GitHub folder-la irukanum
        with open('nb_task.pkl', 'rb') as f: nb_model = pickle.load(f)
        with open('knn_music.pkl', 'rb') as f: knn_model = pickle.load(f)
        with open('encoders.pkl', 'rb') as f: encoders = pickle.load(f)
        return nb_model, knn_model, encoders, True
    except:
        return None, None, None, False

nb_model, knn_model, encoders, is_ml_ready = load_models()

# --- 4. ADVANCED CSS (PINK SIDEBAR & HIGH VISIBILITY) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;900&family=Poppins:wght@400;700;900&display=swap');

    .stApp {
        background: url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
    }
    .main-overlay {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0, 0, 0, 0.82); z-index: -1;
    }

    /* --- SIDEBAR PINK NEON WORDS & ICONS --- */
    [data-testid="stSidebar"] {
        background-color: #050510 !important;
        border-right: 3px solid #ff00c1 !important;
    }
    [data-testid="stSidebar"] span, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #ff00c1 !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 900 !important;
        font-size: 1.15rem !important;
        text-shadow: 0 0 5px rgba(255, 0, 193, 0.5);
    }

    /* Header Design */
    .hero-header {
        text-align: center; padding: 40px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 35px; border: 2px solid #ff00c1;
        backdrop-filter: blur(15px); margin-bottom: 30px;
    }
    .company-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 5rem !important;
        font-weight: 900;
        background: linear-gradient(90deg, #ff00c1, #00d2ff, #92fe9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 12px;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(10, 10, 20, 0.95);
        padding: 30px; border-radius: 25px;
        border: 1px solid rgba(255, 0, 193, 0.4);
        margin-bottom: 25px;
    }

    h2, h3 { color: #00d2ff !important; font-family: 'Orbitron', sans-serif; }
    p, label { font-size: 1.4rem !important; color: white !important; font-family: 'Poppins', sans-serif; font-weight: 600; }

    /* Action Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #ff00c1, #00d2ff);
        color: white !important;
        border-radius: 50px;
        padding: 15px 45px;
        font-weight: 900;
        width: 100%;
        border: none;
        box-shadow: 0 0 30px rgba(255, 0, 193, 0.4);
    }
    </style>
    <div class="main-overlay"></div>
    """, unsafe_allow_html=True)

# --- 5. AUDIO & TEXT AI LOGIC ---
def voice_to_music(audio, sr):
    f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = np.nan_to_num(f0)
    phase = np.cumsum(2 * np.pi * f0 / sr)
    music = 0.5 * np.sin(phase) + 0.3 * np.sin(2 * phase)
    return music / (np.max(np.abs(music)) + 1e-6)

def text_to_melody(text):
    sr = 44100
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration))
    freq = (sum([ord(c) for c in text]) % 400) + 200
    melody = 0.5 * np.sin(2 * np.pi * freq * t)
    return melody, sr

# --- 6. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h2 style='text-align:center; color:#00d2ff !important;'>TECHNOVA</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=120)
    st.write("---")
    
    # Status Check for ML Files
    if is_ml_ready:
        st.success("✅ AI ENGINE: ACTIVE")
    else:
        st.error("⚠️ ML FILES MISSING")

    menu = st.radio(
        "SELECT MODULE:",
        ["🏠 Dashboard", "🧠 Mood AI (ML)", "🎙️ Creative Studio", "♿ Hearing Assist"]
    )
    st.write("---")
    st.markdown("<p style='text-align:center; font-size:0.9rem !important;'>Premium v5.0 Pro</p>", unsafe_allow_html=True)

# --- 7. TOP HEADER ---
st.markdown("""
    <div class="hero-header">
        <h1 class="company-title">TECHNOVA SOLUTION</h1>
        <p style="letter-spacing: 6px; color:#92fe9d; font-size:1.6rem; font-weight:700;">SONICSENSE ULTRA PRO</p>
    </div>
    """, unsafe_allow_html=True)

# --- 8. MODULES ---

# --- DASHBOARD ---
if "Dashboard" in menu:
    st.snow()
    col1, col2 = st.columns([1.6, 1])
    with col1:
        st.markdown("<div class='glass-card'><h2>Technova SonicSense</h2><p>Experience the next generation of sound intelligence. From ML-based mood prediction to creative AI music synthesis, we bridge the gap between human senses and AI technology.</p></div>", unsafe_allow_html=True)
    with col2:
        st.image("https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=500&h=500&fit=crop", use_container_width=True)

# --- MOOD SPOTIFY AI (PDF ML & NO-ERROR LINK) ---
elif "Mood AI" in menu:
    st.markdown("<div class='glass-card'><h3>🧠 AI Mood & Spotify Intelligence</h3></div>", unsafe_allow_html=True)
    
    # Stable Spotify Links for ML Prediction
    playlist_map = {
        "Lo-Fi": "https://open.spotify.com/playlist/37i9dQZF1DX8UebicO9uaR",
        "Electronic": "https://open.spotify.com/playlist/37i9dQZF1DX6J5NfMJS675",
        "Jazz": "https://open.spotify.com/playlist/37i9dQZF1DXbITWG1ZUBIB",
        "Classical": "https://open.spotify.com/playlist/37i9dQZF1DX8u97vXmZp9v",
        "Pop": "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGvYBM3s",
        "Ambient": "https://open.spotify.com/playlist/37i9dQZF1DX3YSRmBhyV9O",
        "Rock": "https://open.spotify.com/playlist/37i9dQZF1DX8FwnS9Y9v9v"
    }

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        # Inputs from your PDF logic
        u_mood = st.selectbox("Current Mood:", ["Calm", "Stressed", "Energetic", "Sad"])
        u_act = st.selectbox("Activity:", ["Studying", "Coding", "Workout", "Relaxing", "Sleeping"])
        
        if st.button("🚀 PREDICT & LAUNCH"):
            if is_ml_ready:
                # ML Prediction Logic from PDF
                m_enc = encoders['le_mood'].transform([u_mood])[0]
                a_enc = encoders['le_activity'].transform([u_act])[0]
                X = np.array([[m_enc, a_enc, datetime.now().hour, 0]]) # TimeOfDay auto-calc
                
                st.session_state.pred_task = encoders['le_task'].inverse_transform(nb_model.predict(X))[0]
                st.session_state.vibe_name = encoders['le_music'].inverse_transform(knn_model.predict(X))[0]
                st.balloons()
            else:
                # Manual Fallback if ML files are missing
                st.session_state.pred_task = "Focus on Work"
                st.session_state.vibe_name = "Lo-Fi"
            st.snow()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if st.session_state.vibe_name:
            vibe = st.session_state.vibe_name
            spotify_url = playlist_map.get(vibe, "https://open.spotify.com/playlist/37i9dQZF1DX8UebicO9uaR")
            
            st.markdown(f"""
                <div class='glass-card' style='text-align:center; border: 2px solid #1DB954;'>
                    <h3>AI Recommendation</h3>
                    <p>Task: <b style='color:#92fe9d;'>{st.session_state.pred_task}</b></p>
                    <p>Playlist Vibe: <b style='color:#00d2ff;'>{vibe}</b></p>
                    <br>
                    <a href="{spotify_url}" target="_blank" style="text-decoration:none;">
                        <button style="background:#1DB954; color:white; border:none; padding:15px 35px; border-radius:50px; font-weight:bold; width:100%; cursor:pointer;">
                            🎧 OPEN PLAYLIST IN SPOTIFY
                        </button>
                    </a>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.image("https://images.unsplash.com/photo-1493225255756-d9584f8606e9?q=80&w=800", use_container_width=True)

# --- CREATIVE STUDIO (3-IN-1: RECORD, UPLOAD, TEXT TO SONG) ---
elif "Creative Studio" in menu:
    st.markdown("<div class='glass-card'><h3>🎙️ Creative AI Studio</h3></div>", unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["🎤 RECORD VOICE", "📤 UPLOAD FILE", "✍️ TEXT TO MELODY"])
    
    with t1:
        st.write("On-the-spot Voice to Music:")
        voice = st.audio_input("Microphone")
        if voice and st.button("✨ TRANSFORM RECORDING"):
            y, sr = librosa.load(voice)
            st.audio(voice_to_music(y, sr), sample_rate=sr)
            st.balloons()

    with t2:
        st.write("Upload MP3/WAV file:")
        up = st.file_uploader("Choose file", type=["mp3","wav"])
        if up and st.button("🚀 TRANSFORM UPLOAD"):
            y, sr = librosa.load(up)
            st.audio(voice_to_music(y, sr), sample_rate=sr)
            st.balloons()

    with t3:
        st.write("Text message to Melody:")
        txt = st.text_input("Enter text (e.g. Technova Pro)")
        if txt and st.button("🎵 GENERATE MELODY"):
            mel, sr_mel = text_to_melody(txt)
            st.audio(mel, sample_rate=sr_mel)
            st.balloons()

# --- HEARING ASSIST (FREQUENCY SHIFT) ---
elif "Hearing Assist" in menu:
    st.markdown("<div class='glass-card'><h3>♿ Inclusive Hearing Assist</h3><p>Optimizing frequencies (Low Pitch) for the hearing impaired.</p></div>", unsafe_allow_html=True)
    up_h = st.file_uploader("Upload audio for assist", type=["mp3", "wav"])
    if up_h:
        y, sr = librosa.load(up_h)
        shift = st.slider("Select Sensitivity (Lower pitch = more vibration)", -12, 0, -8)
        if st.button("🔊 OPTIMIZE Pattern"):
            st.snow()
            y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift)
            st.audio(y_shift * 1.5, sample_rate=sr)
            st.success("Sound optimized for Earspots vibrations.")
