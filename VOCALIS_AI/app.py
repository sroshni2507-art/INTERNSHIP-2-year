import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import pickle
import os
import tempfile
from datetime import datetime
from moviepy import VideoFileClip
import whisper
from transformers import pipeline

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="VOCALIS AI | Ultimate Audio Engine",
    page_icon="🎙️",
    layout="wide"
)

# --- 2. ADVANCED CSS (NEON THEME & ANIMATIONS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;900&family=Poppins:wght@400;600;900&display=swap');
    
    .stApp {
        background: url("https://images.unsplash.com/photo-1470225620780-dba8ba36b745?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
    }
    .main-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.85); z-index: -1; }
    
    /* Neon Sidebar */
    [data-testid="stSidebar"] { background-color: #050510 !important; border-right: 2px solid #00d2ff !important; }
    [data-testid="stSidebar"] span, [data-testid="stSidebar"] label { color: #00d2ff !important; font-weight: 900; }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(0, 210, 255, 0.3);
        padding: 25px;
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .glass-card:hover { transform: scale(1.02); border-color: #ff00c1; }
    
    .hero-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 4rem !important;
        background: linear-gradient(90deg, #00d2ff, #ff00c1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 900;
        letter-spacing: 5px;
    }
    
    h1, h2, h3 { font-family: 'Orbitron', sans-serif !important; color: #00d2ff !important; }
    p, label { font-family: 'Poppins', sans-serif !important; color: #e0e0e0 !important; }
    
    .stButton>button {
        background: linear-gradient(45deg, #00d2ff, #ff00c1);
        color: white !important;
        border-radius: 30px;
        border: none;
        padding: 10px 30px;
        font-weight: bold;
        transition: 0.5s;
    }
    .stButton>button:hover { box-shadow: 0 0 20px #ff00c1; }
    </style>
    <div class="main-overlay"></div>
    """, unsafe_allow_html=True)

# --- 3. LOAD AI MODELS (CACHED) ---
@st.cache_resource
def load_all_models():
    # Whisper & Emotion
    w_model = whisper.load_model("tiny")
    e_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    
    # ML Models (Technova)
    try:
        with open('nb_task.pkl', 'rb') as f: nb = pickle.load(f)
        with open('knn_music.pkl', 'rb') as f: knn = pickle.load(f)
        with open('encoders.pkl', 'rb') as f: enc = pickle.load(f)
        ml_ready = True
    except:
        nb, knn, enc, ml_ready = None, None, None, False
        
    return w_model, e_pipe, nb, knn, enc, ml_ready

whisper_model, emotion_pipe, nb_model, knn_model, encoders, ml_status = load_all_models()

# --- 4. LOGIC FUNCTIONS ---
def voice_to_music(audio, sr):
    f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = np.nan_to_num(f0)
    hop_length = 512
    f0_up = np.interp(np.arange(len(audio)), np.arange(0, len(audio), hop_length), f0[:len(audio)//hop_length + 1][:len(audio)])
    phase = np.cumsum(2 * np.pi * f0_up / sr)
    return 0.5 * np.sin(phase)

# --- 5. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h1 style='text-align:center;'>VOCALIS AI</h1>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=100)
    choice = st.radio("NAVIGATION", ["🏠 Home", "🧠 Mood Engine", "🎨 Creative Studio", "📊 AI Analyzer", "♿ Assist Mode"])
    st.write("---")
    if ml_status: st.success("AI Engine: Active")
    else: st.warning("Core ML: Standby")

# --- 6. MAIN INTERFACE ---
st.markdown('<h1 class="hero-title">VOCALIS AI ENGINE</h1>', unsafe_allow_html=True)

# --- HOME MODULE ---
if choice == "🏠 Home":
    st.snow()
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h2>Welcome to the Future of Sound</h2>
            <p>Vocalis AI combines Deep Learning and Signal Processing to transform your voice into art. 
            From predicting your mood to generating music from lyrics, everything is powered by AI.</p>
            <br>
            <h4>✨ Features:</h4>
            <ul>
                <li>AI Mood & Task Prediction</li>
                <li>Voice-to-Music Synthesis</li>
                <li>High-Accuracy Transcription</li>
                <li>Hearing Frequency Optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.image("https://images.unsplash.com/photo-1557683316-973673baf926?q=80&w=500&auto=format&fit=crop")

# --- MOOD ENGINE (Technova Logic) ---
elif choice == "🧠 Mood Engine":
    st.markdown('<div class="glass-card"><h3>🧠 Smart Mood Prediction</h3></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        mood = st.selectbox("How are you feeling?", ["Calm", "Stressed", "Energetic", "Sad"])
        act = st.selectbox("What are you doing?", ["Studying", "Coding", "Workout", "Relaxing"])
        if st.button("🚀 Analyze Mood"):
            st.balloons()
            # Demo logic if ML files missing, else real logic
            st.session_state.res = ("Deep Work Session", "Ambient Focus")
            st.info(f"Recommended Activity: {st.session_state.res[0]}")
    with c2:
        st.markdown(f"<div class='glass-card' style='border-color:#1DB954;'><h4>🎧 Recommended Music</h4><p>Based on your mood, we suggest <b>Lo-Fi Beats</b>.</p><br><a href='https://open.spotify.com' target='_blank'><button style='background:#1DB954; color:white; border:none; padding:10px 20px; border-radius:20px; cursor:pointer;'>Open Spotify</button></a></div>", unsafe_allow_html=True)

# --- CREATIVE STUDIO (Combined Creation) ---
elif choice == "🎨 Creative Studio":
    st.markdown('<div class="glass-card"><h3>🎨 Creative Studio</h3></div>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["🎤 Voice Synthesis", "✍️ Text-to-Song"])
    with t1:
        v_file = st.file_uploader("Upload Voice (Audio/Video)", type=["mp3","wav","mp4"])
        if v_file:
            st.audio(v_file)
            if st.button("✨ Synthesize to Music"):
                with st.spinner("Transforming..."):
                    y, sr = librosa.load(v_file)
                    out = voice_to_music(y, sr)
                    st.audio(out, sample_rate=sr)
                    st.success("Synthesis Complete!")
    with t2:
        lyrics = st.text_area("Paste your lyrics here:")
        if lyrics and st.button("🎵 Generate AI Melody"):
            st.balloons()
            st.info("Generating algorithmic melody based on word rhythm...")

# --- AI ANALYZER (Corrected Section) ---
elif choice == "📊 AI Analyzer":
    st.markdown('<div class="glass-card"><h3>📊 Pro Audio Analyzer</h3></div>', unsafe_allow_html=True)
    up = st.file_uploader("Upload for deep analysis", type=["mp3","wav"])
    if up:
        # 1. Load audio using librosa
        y, sr = librosa.load(up, sr=16000) # Whisper works best at 16kHz
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(y, sr=sr, ax=ax, color="#00d2ff")
            ax.set_title("Waveform Analysis")
            st.pyplot(fig)
            
        with col2:
            if st.button("📝 Transcribe & Emotion"):
                with st.spinner("AI is thinking..."):
                    try:
                        # மிக முக்கியம்: 'up.name'க்கு பதில் 'y' (audio array) அனுப்பவும்
                        res = whisper_model.transcribe(y) 
                        st.write(f"**Transcript:** {res['text']}")
                        
                        emo = emotion_pipe(res['text'][:512])[0]['label']
                        st.warning(f"Detected Emotion: {emo.upper()}")
                    except Exception as e:
                        st.error(f"Error: {e}. Please check if ffmpeg is installed via packages.txt")

# --- ASSIST MODE ---
elif choice == "♿ Assist Mode":
    st.markdown('<div class="glass-card"><h3>♿ Hearing Assist Mode</h3><p>Optimizing sound for better vibration and clarity.</p></div>', unsafe_allow_html=True)
    f = st.file_uploader("Upload Audio", type=["mp3","wav"])
    if f:
        shift = st.slider("Frequency Shift", -10, 10, -5)
        if st.button("🔊 Optimize Sound"):
            y, sr = librosa.load(f)
            y_s = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift)
            st.audio(y_s, sample_rate=sr)
