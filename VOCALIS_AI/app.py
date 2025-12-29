import streamlit as st
import librosa
import librosa.display
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import soundfile as sf
import tempfile
import time
import requests
import io
from datetime import datetime
from streamlit_lottie import st_lottie
from pydub import AudioSegment

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SONICSENSE ULTRA PRO | Technova x Vocalis",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. LOTTIE ANIMATION LOADER ---
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200: return None
        return r.json()
    except: return None

lottie_ai = load_lottieurl("https://lottie.host/804d9c75-3432-4770-8777-628f800c01a5/eH6F1X9K3L.json") 
lottie_music = load_lottieurl("https://lottie.host/83e0e788-779d-4033-9092-22538965873a/vX6yUf0wV8.json")

# --- 3. SESSION STATE INITIALIZATION ---
if 'pred_task' not in st.session_state: st.session_state.pred_task = None
if 'pred_genre' not in st.session_state: st.session_state.pred_genre = None
if 'taps' not in st.session_state: st.session_state.taps = []
if 'bpm_val' not in st.session_state: st.session_state.bpm_val = 0

# --- 4. SMART PATH LOGIC FOR ML FILES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
@st.cache_resource
def load_models():
    nb_path = os.path.join(BASE_DIR, 'nb_task.pkl')
    knn_path = os.path.join(BASE_DIR, 'knn_music.pkl')
    enc_path = os.path.join(BASE_DIR, 'encoders.pkl')
    if os.path.exists(nb_path) and os.path.exists(knn_path) and os.path.exists(enc_path):
        try:
            with open(nb_path, 'rb') as f: nb_model = pickle.load(f)
            with open(knn_path, 'rb') as f: knn_model = pickle.load(f)
            with open(enc_path, 'rb') as f: encoders = pickle.load(f)
            return nb_model, knn_model, encoders, True
        except: return None, None, None, False
    else: return None, None, None, False

nb_model, knn_model, encoders, is_ml_ready = load_models()

# --- 5. ADVANCED CSS (PINK NEON STYLE) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;900&family=Poppins:wght@400;700;900&display=swap');
    
    .stApp { background: url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop"); background-size: cover; background-attachment: fixed; }
    .main-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.85); z-index: -1; }
    
    /* Sidebar Neon Pink Styling */
    [data-testid="stSidebar"] { background-color: #050510 !important; border-right: 3px solid #ff00c1 !important; }
    [data-testid="stSidebar"] span, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #ff00c1 !important; font-family: 'Poppins', sans-serif !important; font-weight: 900 !important; font-size: 1.1rem !important;
    }

    .hero-header { text-align: center; padding: 30px; background: rgba(255, 255, 255, 0.05); border-radius: 35px; border: 2px solid #ff00c1; backdrop-filter: blur(15px); margin-bottom: 20px; }
    .company-title { 
        font-family: 'Orbitron', sans-serif; font-size: 3.5rem !important; font-weight: 900; 
        background: linear-gradient(90deg, #ff00c1, #00d2ff, #92fe9d); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
        letter-spacing: 5px;
    }
    
    .glass-card { 
        background: rgba(10, 10, 20, 0.95); padding: 25px; border-radius: 20px; 
        border: 1px solid rgba(255, 0, 193, 0.4); margin-bottom: 20px; 
    }
    h2, h3 { color: #00d2ff !important; font-family: 'Orbitron', sans-serif; }
    p, label { color: white !important; font-family: 'Poppins', sans-serif; font-weight: 600; }

    .stButton>button { 
        background: linear-gradient(45deg, #ff00c1, #00d2ff); color: white !important; 
        border-radius: 50px; font-weight: 900; transition: 0.5s; border: none; width: 100%;
    }
    </style>
    <div class="main-overlay"></div>
    """, unsafe_allow_html=True)

# --- 6. LOGIC FUNCTIONS ---

def voice_to_music(audio, sr):
    hop_length = 512
    f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), hop_length=hop_length)
    f0 = np.nan_to_num(f0)
    phase = np.cumsum(2 * np.pi * np.repeat(f0, hop_length)[:len(audio)] / sr)
    music = 0.5 * np.sin(phase)
    return music

def voice_morpher(y, sr, effect):
    if effect == "Child 👶": return librosa.effects.pitch_shift(y, sr=sr, n_steps=5)
    if effect == "Villain 👿": return librosa.effects.pitch_shift(y, sr=sr, n_steps=-5)
    if effect == "Robot 🤖": return np.clip(librosa.effects.pitch_shift(y, sr=sr, n_steps=2) * 1.5, -1, 1)
    return y

def text_to_song_logic(text):
    sr = 44100
    words = text.split()
    full_song = np.array([])
    scale = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25] 
    for word in words:
        freq = scale[len(word) % len(scale)]
        t = np.linspace(0, 0.5, int(sr * 0.5))
        note = 0.5 * np.sin(2 * np.pi * freq * t) * np.exp(-3 * t / 0.5)
        full_song = np.concatenate([full_song, note])
    return full_song, sr

def lyric_assistant(theme):
    lyrics = {"Love": ["Heart skips a beat", "Stars align for us"], "Space": ["Floating in dark", "Galaxy of dreams"], "Rain": ["Droplets on window", "Wash away pain"]}
    return lyrics.get(theme, ["Sing along..."])

# --- 7. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>TECHNOVA x VOCALIS</h2>", unsafe_allow_html=True)
    if lottie_ai: st_lottie(lottie_ai, height=120, key="nav_ai")
    st.write("---")
    choice = st.radio("SELECT ENGINE:", ["🏠 Dashboard", "🧠 Mood AI (ML)", "🎭 Creative Studio", "♿ Assist Mode", "🎹 BPM Tapper"])
    st.write("---")
    if is_ml_ready: st.success("✅ AI ENGINE: ACTIVE")
    else: st.error("⚠️ ML CORE MISSING")

# --- HEADER ---
st.markdown("""<div class="hero-header"><h1 class="company-title">SONICSENSE ULTRA PRO</h1><p style="color:#92fe9d; font-weight:700; letter-spacing:3px;">HYBRID AUDIO FUSION ENGINE</p></div>""", unsafe_allow_html=True)

# --- 8. MODULES ---

# --- DASHBOARD ---
if choice == "🏠 Dashboard":
    st.snow()
    col1, col2 = st.columns([1.5, 1])
    with col1: 
        st.markdown("<div class='glass-card'><h2>The Future is Audio</h2><p>Experience the synergy of Machine Learning and Signal Processing. Analyze moods, create music, and assist the world with Technova's SonicSense.</p></div>", unsafe_allow_html=True)
    with col2: 
        if lottie_music: st_lottie(lottie_music, height=250, key="dash_music")

# --- MOOD AI ---
elif choice == "🧠 Mood AI (ML)":
    st.markdown("<div class='glass-card'><h3>🧠 AI Mood & Task Prediction</h3></div>", unsafe_allow_html=True)
    genre_map = {"Lo-Fi": "lofi focus music", "Electronic": "electronic workout", "Jazz": "smooth jazz", "Pop": "top hits"}
    col1, col2 = st.columns([1, 1.2])
    with col1:
        u_mood = st.selectbox("Current Mood:", ["Calm", "Stressed", "Energetic", "Sad"])
        u_act = st.selectbox("Activity:", ["Studying", "Coding", "Workout", "Relaxing"])
        if st.button("🚀 PREDICT"):
            if is_ml_ready:
                m_enc = encoders['le_mood'].transform([u_mood])[0]
                a_enc = encoders['le_activity'].transform([u_act])[0]
                st.session_state.pred_task = encoders['le_task'].inverse_transform(nb_model.predict([[m_enc, a_enc, datetime.now().hour, 0]]))[0]
                st.session_state.pred_genre = encoders['le_music'].inverse_transform(knn_model.predict([[m_enc, a_enc, datetime.now().hour, 0]]))[0]
                st.balloons()
            else:
                st.session_state.pred_task = "Focus Session"; st.session_state.pred_genre = "Lo-Fi"
    with col2:
        if st.session_state.pred_genre:
            st.markdown(f"<div class='glass-card'><h3>🎧 Recommendation</h3><p>Task: {st.session_state.pred_task}</p><p>Music: {st.session_state.pred_genre}</p></div>", unsafe_allow_html=True)

# --- CREATIVE STUDIO ---
elif choice == "🎭 Creative Studio":
    st.markdown("<div class='glass-card'><h3>🎨 Creative AI Studio</h3></div>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎭 VOICE MORPH", "🎤 CONVERT", "✍️ LYRICS", "🎵 MIXER", "🎼 VISUALS"])
    
    with tab1:
        v_m = st.file_uploader("Upload Voice:", type=["wav", "mp3"], key="morph")
        eff = st.selectbox("Character:", ["Child 👶", "Villain 👿", "Robot 🤖"])
        if v_m and st.button("✨ TRANSFORM MORPH"):
            y, sr = librosa.load(v_m); morphed = voice_morpher(y, sr, eff)
            st.audio(morphed, sample_rate=sr)
            
    with tab2:
        v_rec = st.audio_input("Record or Upload for Voice-to-Music:")
        if v_rec and st.button("🚀 CONVERT TO MUSIC"):
            y, sr = librosa.load(v_rec); st.audio(voice_to_music(y, sr), sample_rate=sr)
            
    with tab3:
        lyrics_txt = st.text_area("Write lyrics for a theme:")
        if lyrics_txt and st.button("🎵 GENERATE SONG"):
            song, sr_s = text_to_song_logic(lyrics_txt); st.audio(song, sample_rate=sr_s)
            
    with tab4:
        v_file = st.file_uploader("Voice File:", type=["mp3"], key="v_mix")
        b_file = st.file_uploader("BGM File:", type=["mp3"], key="b_mix")
        if v_file and b_file and st.button("🎚️ MIX AUDIO"):
            v_d, sr = librosa.load(v_file); b_d, _ = librosa.load(b_file, sr=sr)
            mix = v_d[:min(len(v_d), len(b_d))] + (b_d[:min(len(v_d), len(b_d))] * 0.2)
            st.audio(mix, sample_rate=sr)
            
    with tab5:
        vis_up = st.file_uploader("Visualize Audio:", type=["mp3"], key="vis")
        if vis_up:
            y, sr = librosa.load(vis_up); fig, ax = plt.subplots(figsize=(10, 3))
            librosa.display.waveshow(y, sr=sr, ax=ax, color="#ff00c1")
            ax.set_facecolor('black'); fig.patch.set_facecolor('black'); st.pyplot(fig)

# --- ASSIST MODE ---
elif choice == "♿ Assist Mode":
    st.markdown("<div class='glass-card'><h3>♿ Hearing Assist</h3><p>Frequency optimization for enhanced vibrations.</p></div>", unsafe_allow_html=True)
    h_up = st.file_uploader("Upload Audio:", type=["mp3"])
    if h_up and st.button("🔊 OPTIMIZE FOR VIBRATION"):
        y, sr = librosa.load(h_up); out = librosa.effects.pitch_shift(y, sr=sr, n_steps=-8)
        st.audio(out * 1.5, sample_rate=sr)

# --- BPM TAPPER ---
elif choice == "🎹 BPM Tapper":
    st.markdown("<div class='glass-card'><h3>🎹 BPM Tapper</h3></div>", unsafe_allow_html=True)
    if st.button("🥁 TAP BEAT", use_container_width=True):
        st.session_state.taps.append(time.time())
        if len(st.session_state.taps) > 1:
            st.session_state.bpm_val = 60 / np.mean(np.diff(st.session_state.taps[-8:]))
    st.metric("Detected BPM", f"{int(st.session_state.bpm_val)}")
    if st.button("🔄 RESET"):
        st.session_state.taps = []; st.session_state.bpm_val = 0; st.rerun()
