import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import io

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="TECHNOVA SOLUTION | SonicSense Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PREMIUM CSS (PINK SIDEBAR & VISIBILITY) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;900&family=Poppins:wght@400;700&display=swap');

.stApp {
    background: url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop");
    background-size: cover;
    background-attachment: fixed;
}
.main-overlay {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0, 0, 0, 0.8); z-index: -1;
}

/* SIDEBAR PINK STYLE */
[data-testid="stSidebar"] {
    background: #050510 !important;
    border-right: 2px solid #ff00c1 !important;
}
[data-testid="stSidebar"] * {
    color: #ff00c1 !important;
    font-family: 'Poppins', sans-serif;
    font-weight: 800 !important;
}

/* Glassmorphic Cards */
.glass {
    background: rgba(10, 10, 20, 0.92);
    padding: 30px;
    border-radius: 25px;
    border: 1px solid rgba(0, 210, 255, 0.3);
    margin-bottom: 25px;
}

.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 4.5rem !important;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #ff00c1, #00d2ff, #92fe9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 10px;
}

h2, h3 { color: #00d2ff !important; font-family: 'Orbitron', sans-serif; }
p, label { font-size: 1.3rem !important; color: white !important; font-family: 'Poppins', sans-serif; }

.stButton>button {
    background: linear-gradient(45deg, #ff00c1, #00d2ff);
    color: white !important;
    border-radius: 50px;
    padding: 12px 40px;
    font-weight: 900;
    width: 100%;
}
</style>
<div class="main-overlay"></div>
""", unsafe_allow_html=True)

# --- 3. CORE LOGIC FUNCTIONS ---

def voice_to_music(audio, sr):
    f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = np.nan_to_num(f0)
    phase = np.cumsum(2 * np.pi * f0 / sr)
    music = 0.5 * np.sin(phase) + 0.3 * np.sin(2 * phase)
    return music / (np.max(np.abs(music)) + 1e-6)

def text_to_music(text):
    sr = 22050
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration))
    # Unique frequency based on text
    freq = (sum([ord(c) for c in text]) % 400) + 200
    melody = 0.5 * np.sin(2 * np.pi * freq * t)
    return melody, sr

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>TECHNOVA</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=100)
    st.write("---")
    menu = st.radio("NAVIGATE", ["🏠 Dashboard", "🧠 Mood Spotify AI", "🎙️ Creative Studio", "♿ Hearing Assist"])
    st.write("---")
    st.success("⚡ AI ENGINE : ONLINE")

# --- 5. HEADER ---
st.markdown("<h1 class='hero-title'>TECHNOVA SOLUTION</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#92fe9d; font-weight:bold;'>SONICSENSE ULTRA PRO</p>", unsafe_allow_html=True)

# --- 6. MODULES ---

# --- DASHBOARD ---
if menu == "🏠 Dashboard":
    st.snow()
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("<div class='glass'><h2>Next-Gen Audio Intelligence</h2><p>Technova Solution bridges the gap between sound and AI. We create accessible audio tools for everyone.</p></div>", unsafe_allow_html=True)
    with col2:
        st.image("https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=600&fit=crop", use_container_width=True)

# --- MOOD SPOTIFY AI (ERROR FIXED) ---
elif menu == "🧠 Mood Spotify AI":
    st.markdown("<div class='glass'><h3>🧠 Mood-Based Smart Suggestion</h3></div>", unsafe_allow_html=True)
    
    mood_map = {
        "Energetic 🔥": "37i9dQZF1DX76W9SwwE6v4",
        "Calm 🌊": "37i9dQZF1DX8UebicO9uaR",
        "Focused 🎯": "37i9dQZF1DX4sWSp4sm94f",
        "Stressed 🧘": "37i9dQZF1DX3YSRmBhyV9O",
        "Devotional ✨": "37i9dQZF1DX0S69v9S94G0"
    }

    c1, c2 = st.columns([1, 1.5])
    with c1:
        mood_choice = st.selectbox("HOW ARE YOU FEELING?", list(mood_map.keys()))
        if st.button("🚀 LAUNCH SESSION"):
            st.session_state.active_mood = mood_choice
            st.balloons()
            st.snow()

    with c2:
        if 'active_mood' in st.session_state:
            pid = mood_map[st.session_state.active_mood]
            embed_url = f"https://open.spotify.com/embed/playlist/{pid}?utm_source=generator&theme=0"
            components.iframe(embed_url, height=450, scrolling=False)
        else:
            st.info("Choose a mood to see recommendations.")

# --- CREATIVE STUDIO (NEW FEATURES) ---
elif menu == "🎙️ Creative Studio":
    st.markdown("<div class='glass'><h3>🎙️ Creative AI Studio</h3></div>", unsafe_allow_html=True)
    
    t1, t2, t3 = st.tabs(["🎤 LIVE RECORD", "📤 UPLOAD FILE", "✍️ TEXT TO SONG"])
    
    with t1:
        voice = st.audio_input("Record your voice now")
        if voice and st.button("✨ TRANSFORM LIVE VOICE"):
            y, sr = librosa.load(voice)
            out = voice_to_music(y, sr)
            st.audio(out, sample_rate=sr)
            st.balloons()

    with t2:
        up = st.file_uploader("Upload Audio", type=["wav", "mp3"])
        if up and st.button("🚀 CONVERT UPLOAD"):
            y, sr = librosa.load(up)
            out = voice_to_music(y, sr)
            st.audio(out, sample_rate=sr)
            st.balloons()

    with t3:
        txt = st.text_input("Enter text to generate melody")
        if txt and st.button("🎵 GENERATE TUNE"):
            mel, sr_mel = text_to_music(txt)
            st.audio(mel, sample_rate=sr_mel)
            st.balloons()

# --- HEARING ASSIST (FREQUENCY CHANGE) ---
elif menu == "♿ Hearing Assist":
    st.markdown("<div class='glass'><h3>♿ Inclusive Hearing Assist</h3><p>Optimizing frequencies for the hearing impaired.</p></div>", unsafe_allow_html=True)
    
    h_file = st.file_uploader("Upload Audio for Frequency Optimization", type=["wav", "mp3"])
    if h_file:
        y, sr = librosa.load(h_file)
        
        # User chooses the shift (High pitch loss is common, so we shift down to low/bass)
        shift = st.slider("Adjust Frequency (Lower = Better for feeling vibrations)", -12, 0, -6)
        
        if st.button("🔊 OPTIMIZE FOR HEARING"):
            st.snow()
            # PITCH SHIFTING LOGIC
            y_optimized = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift)
            
            # Boost Volume/Amplitude for better haptic/bone conduction
            y_optimized = y_optimized * 1.5
            y_optimized = np.clip(y_optimized, -1.0, 1.0)
            
            st.success("Frequency Adjusted! Optimized for Hearing Aids & Bone Conduction.")
            st.audio(y_optimized, sample_rate=sr)
            
            # Pulse Graph for visualization
            st.line_chart(y_optimized[:10000])
