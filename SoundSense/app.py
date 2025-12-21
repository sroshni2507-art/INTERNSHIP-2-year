import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import io

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TECHNOVA SOLUTION | Pro AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. SESSION STATE INITIALIZATION (FIXES ATTRIBUTE ERROR) ---
if 'mood_selected' not in st.session_state:
    st.session_state.mood_selected = "Calm 🌊"

# --- 3. PREMIUM CSS (PINK SIDEBAR & BOLD VISIBILITY) ---
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
    /* Force Sidebar Words & Icons to be PINK */
    [data-testid="stSidebar"] span, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] li {
        color: #ff00c1 !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 900 !important;
        font-size: 1.2rem !important;
        text-shadow: 0 0 5px rgba(255, 0, 193, 0.5);
    }

    /* TECHNOVA HEADER */
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
        color: white;
    }

    h2, h3 { color: #00d2ff !important; font-family: 'Orbitron', sans-serif; font-size: 2.7rem !important; }
    p, label { font-size: 1.5rem !important; color: white !important; font-family: 'Poppins', sans-serif; font-weight: 600; }

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

# --- 4. CORE AI LOGIC ---
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
    # Unique frequency based on text content
    freq = (sum([ord(c) for c in text]) % 400) + 200
    melody = 0.5 * np.sin(2 * np.pi * freq * t)
    return melody, sr

# --- 5. SIDEBAR NAVIGATION (PINK WORDS & ICONS) ---
with st.sidebar:
    st.markdown("<h2 style='text-align:center; color:#00d2ff !important;'>TECHNOVA</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=120)
    st.write("---")
    
    # Selection Menu with Icons
    menu = st.radio(
        "SELECT MODULE:",
        ["🏠 Dashboard", "🧠 Mood Spotify AI", "🎙️ Creative Studio", "♿ Hearing Assist"]
    )
    st.write("---")
    st.success("⚡ AI ENGINE: ONLINE")

# --- 6. TOP HEADER ---
st.markdown("""
    <div class="hero-header">
        <h1 class="company-title">TECHNOVA SOLUTION</h1>
        <p style="letter-spacing: 6px; color:#92fe9d; font-size:1.6rem; font-weight:700;">SONICSENSE ULTRA PRO</p>
    </div>
    """, unsafe_allow_html=True)

# --- 7. MODULES ---

# --- DASHBOARD ---
if "Dashboard" in menu:
    st.snow()
    col1, col2 = st.columns([1.6, 1])
    with col1:
        st.markdown("<div class='glass-card'><h2>Technova Dashboard</h2><p>Welcome to Technova Solution. Bridge the gap between AI and sound. Explore creative tools and accessibility features designed for everyone.</p></div>", unsafe_allow_html=True)
    with col2:
        st.image("https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=500&h=500&fit=crop", use_container_width=True)

# --- MOOD SPOTIFY AI (NO ERROR VERSION) ---
elif "Mood Spotify AI" in menu:
    st.markdown("<div class='glass-card'><h3>🧠 Mood-Based Spotify AI</h3></div>", unsafe_allow_html=True)
    
    # Direct Spotify Links for each mood (Works 100% - No Page Not Found)
    mood_link_map = {
        "Energetic 🔥": "https://open.spotify.com/playlist/37i9dQZF1DX76W9SwwE6v4",
        "Calm 🌊": "https://open.spotify.com/playlist/37i9dQZF1DX8UebicO9uaR",
        "Focused 🎯": "https://open.spotify.com/playlist/37i9dQZF1DX4sWSp4sm94f",
        "Stressed 🧘": "https://open.spotify.com/playlist/37i9dQZF1DX3YSRmBhyV9O",
        "Devotional ✨": "https://open.spotify.com/playlist/37i9dQZF1DX0S69v9S94G0"
    }

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("### How are you feeling?")
        u_mood = st.selectbox("CHOOSE YOUR VIBE:", list(mood_link_map.keys()))
        
        # Action Button
        st.write("---")
        st.write(f"Click the button below to open **{u_mood}** on Spotify.")
        
        # Direct Link Button (This solves the Error permanently)
        spotify_url = mood_link_map[u_mood]
        st.link_button(f"🟢 OPEN {u_mood.upper()} ON SPOTIFY", spotify_url, use_container_width=True)
        
        if st.button("🚀 LAUNCH SESSION EFFECTS"):
            st.balloons()
            st.snow()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.image("https://images.unsplash.com/photo-1493225255756-d9584f8606e9?q=80&w=800", use_container_width=True)
        st.markdown("<p style='text-align:center;'>Music helps maintain Flow State. Technova AI recommends background listening for productivity.</p>", unsafe_allow_html=True)

# --- CREATIVE STUDIO (RECORD + UPLOAD + TEXT TO SONG) ---
elif "Creative Studio" in menu:
    st.markdown("<div class='glass-card'><h3>🎙️ Creative AI Studio</h3></div>", unsafe_allow_html=True)
    
    # TABS FOR CLEAN UI
    t1, t2, t3 = st.tabs(["🎤 RECORD LIVE", "📤 UPLOAD FILE", "✍️ TEXT TO SONG"])
    
    with t1:
        st.write("On-the-spot Voice to Music Conversion:")
        voice = st.audio_input("Record your voice")
        if voice and st.button("✨ TRANSFORM RECORDING"):
            y, sr = librosa.load(voice)
            out = voice_to_music(y, sr)
            st.audio(out, sample_rate=sr)
            st.balloons()

    with t2:
        st.write("Upload an audio file (MP3/WAV) to convert:")
        up = st.file_uploader("Choose Audio File", type=["mp3","wav"])
        if up and st.button("🚀 TRANSFORM UPLOAD"):
            y, sr = librosa.load(up)
            out = voice_to_music(y, sr)
            st.audio(out, sample_rate=sr)
            st.balloons()

    with t3:
        st.write("Type text and AI will compose a melody for it:")
        txt = st.text_input("Enter text (e.g., Technova Magic)")
        if txt and st.button("🎵 GENERATE MELODY"):
            st.toast("Technova AI is composing...")
            mel, sr_m = text_to_melody(txt)
            st.audio(mel, sample_rate=sr_m)
            st.balloons()

# --- HEARING ASSIST ---
elif "Hearing Assist" in menu:
    st.markdown("<div class='glass-card'><h3>♿ Inclusive Hearing Assist</h3><p>Optimizing frequencies for hearing aids and bone-conduction devices.</p></div>", unsafe_allow_html=True)
    up_h = st.file_uploader("Upload audio for frequency shift", type=["mp3", "wav"])
    if up_h:
        y, sr = librosa.load(up_h)
        # Shift steps: Low frequency is easier to feel for hearing impaired
        shift = st.slider("Frequency Transposition (Lower = more vibration)", -12, 0, -8)
        if st.button("🔊 OPTIMIZE FOR VIBRATION"):
            st.snow()
            y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift)
            st.audio(y_shift * 1.5, sample_rate=sr)
            st.success("Optimization Complete! Ready for pattern vibration.")
