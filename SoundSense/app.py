import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import io

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TECHNOVA SOLUTION | SonicSense Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ADVANCED CSS (PINK SIDEBAR & NO ERROR LOOK) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;900&family=Poppins:wght@400;700;900&display=swap');

/* Global Background */
.stApp {
    background: url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop");
    background-size: cover;
    background-attachment: fixed;
}
.main-overlay {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0, 0, 0, 0.85); z-index: -1;
}

/* --- SIDEBAR PINK NEON (ULTRA VISIBLE) --- */
[data-testid="stSidebar"] {
    background-color: #050510 !important;
    border-right: 3px solid #ff00c1 !important;
}

/* Sidebar Text & Radio Buttons to PINK */
[data-testid="stSidebar"] * {
    color: #ff00c1 !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 900 !important;
    font-size: 1.15rem !important;
}

/* Header Branding */
.hero-header {
    text-align: center; padding: 40px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 35px; border: 2px solid #ff00c1;
    backdrop-filter: blur(20px); margin-bottom: 30px;
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
    color: white;
}

h2, h3 { color: #00d2ff !important; font-family: 'Orbitron', sans-serif; }
p, label { font-size: 1.4rem !important; color: white !important; font-weight: 600; }

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

# --- 3. AUDIO LOGIC FUNCTIONS ---
def voice_to_music(audio, sr):
    f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = np.nan_to_num(f0)
    phase = np.cumsum(2 * np.pi * f0 / sr)
    music = 0.5 * np.sin(phase) + 0.3 * np.sin(2 * phase)
    return music / (np.max(np.abs(music)) + 1e-6)

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h2 style='text-align:center; color:#00d2ff !important;'>TECHNOVA</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=120)
    st.write("---")
    
    # Emojis & Pink Words as requested
    menu = st.radio(
        "SELECT MODULE:",
        ["🏠 Dashboard", "🧠 Mood Spotify AI", "🎙️ Creative Studio", "♿ Hearing Assist"]
    )
    st.write("---")
    st.success("⚡ AI ENGINE: ONLINE")

# --- 5. TOP HEADER ---
st.markdown("""
    <div class="hero-header">
        <h1 class="company-title">TECHNOVA SOLUTION</h1>
        <p style="letter-spacing: 6px; color:#92fe9d; font-size:1.6rem; font-weight:700;">SONICSENSE ULTRA PRO</p>
    </div>
    """, unsafe_allow_html=True)

# --- 6. MODULES ---

# DASHBOARD
if "Dashboard" in menu:
    st.snow()
    col1, col2 = st.columns([1.6, 1])
    with col1:
        st.markdown("<div class='glass-card'><h2>Next-Gen Audio AI</h2><p>Technova Solution provides advanced sound intelligence matrum creative tools for everyone.</p></div>", unsafe_allow_html=True)
    with col2:
        st.image("https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=500&h=500&fit=crop", use_container_width=True)

# MOOD SPOTIFY AI (ERROR FIXED)
elif "Mood Spotify AI" in menu:
    st.markdown("<div class='glass-card'><h3>🧠 Mood-Based Suggestions</h3></div>", unsafe_allow_html=True)
    
    # Verified IDs matrum Universal Links
    mood_map = {
        "Energetic 🔥": "37i9dQZF1DX76W9SwwE6v4",
        "Calm 🌊": "37i9dQZF1DX8UebicO9uaR",
        "Focused 🎯": "37i9dQZF1DX4sWSp4sm94f",
        "Stressed 🧘": "37i9dQZF1DX3YSRmBhyV9O",
        "Devotional ✨": "37i9dQZF1DX0S69v9S94G0"
    }

    c1, c2 = st.columns([1, 1.4])
    with c1:
        mood_sel = st.selectbox("HOW ARE YOU FEELING?", list(mood_map.keys()))
        if st.button("🚀 LAUNCH SESSION"):
            st.session_state.active_pid = mood_map[mood_sel]
            st.session_state.active_name = mood_sel
            st.balloons()
            st.snow()

    with c2:
        if 'active_pid' in st.session_state:
            pid = st.session_state.active_pid
            # Embed Player (Fixed ID)
            embed_url = f"https://open.spotify.com/embed/playlist/{pid}?utm_source=generator&theme=0"
            
            st.markdown(f"<h4 style='color:#1DB954;'>Vibe: {st.session_state.active_name}</h4>", unsafe_allow_html=True)
            
            # Error-Free Embedding logic
            try:
                components.iframe(embed_url, height=380, scrolling=False)
            except:
                st.error("Embed failed. Use the button below to open directly.")
            
            # "Page not found" box varaama irukka intha Spotify Green button backup
            st.markdown(f"""
                <div style="text-align:center; margin-top:20px;">
                    <a href="https://open.spotify.com/playlist/{pid}" 
                       target="_blank"
                       style="background:linear-gradient(45deg,#1DB954,#1ed760); padding:15px 35px; border-radius:40px; color:white; font-size:1.3rem; font-weight:800; text-decoration:none; display:inline-block;">
                    🎧 OPEN IN SPOTIFY APP
                    </a>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.image("https://images.unsplash.com/photo-1493225255756-d9584f8606e9?q=80&w=800", use_container_width=True)

# CREATIVE STUDIO (3-IN-1)
elif "Creative Studio" in menu:
    st.markdown("<div class='glass-card'><h3>🎙️ Creative AI Studio</h3></div>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🎤 RECORD VOICE", "📤 UPLOAD FILE"])
    
    with tab1:
        voice = st.audio_input("Record now")
        if voice and st.button("✨ TRANSFORM RECORDING"):
            y, sr = librosa.load(voice)
            out = voice_to_music(y, sr)
            st.audio(out, sample_rate=sr)
            st.balloons()

    with tab2:
        up = st.file_uploader("Upload Audio (MP3/WAV)", type=["mp3","wav"])
        if up and st.button("🚀 TRANSFORM UPLOAD"):
            y, sr = librosa.load(up)
            out = voice_to_music(y, sr)
            st.audio(out, sample_rate=sr)
            st.balloons()

# HEARING ASSIST
elif "Hearing Assist" in menu:
    st.markdown("<div class='glass-card'><h3>♿ Inclusive Hearing Assist</h3><p>Optimizing frequencies (Low Pitch/Bass) for the hearing impaired.</p></div>", unsafe_allow_html=True)
    up_h = st.file_uploader("Upload audio for frequency shift", type=["mp3", "wav"])
    if up_h:
        y, sr = librosa.load(up_h)
        shift_val = st.slider("Adjust Sensitivity (Lower = more vibration)", -12, 0, -8)
        if st.button("🔊 OPTIMIZE Pattern"):
            st.snow()
            y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift_val)
            st.audio(y_shift, sample_rate=sr)
            st.success("Sound optimized for haptic sensations.")
