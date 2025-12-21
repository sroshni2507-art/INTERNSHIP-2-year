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

# --- 2. PREMIUM CSS (PINK SIDEBAR & NEON THEME) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;900&family=Poppins:wght@400;700&display=swap');

/* Main Background Overlay */
.stApp {
    background: url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop");
    background-size: cover;
    background-attachment: fixed;
}
.main-overlay {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0, 0, 0, 0.82); z-index: -1;
}

/* SIDEBAR PINK & ICON STYLE */
[data-testid="stSidebar"] {
    background: #050515 !important;
    border-right: 3px solid #ff00c1 !important;
}
[data-testid="stSidebar"] * {
    color: #ff00c1 !important; /* PINK WORDS */
    font-family: 'Poppins', sans-serif;
    font-weight: 800 !important;
    font-size: 1.15rem;
}

/* Glassmorphic Cards */
.glass-card {
    background: rgba(10, 10, 20, 0.95);
    padding: 30px;
    border-radius: 25px;
    border: 1px solid rgba(255, 0, 193, 0.4);
    margin-bottom: 25px;
}

/* Technova Title Style */
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 5rem !important;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #ff00c1, #00d2ff, #92fe9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 12px;
}

h2, h3 { color: #00d2ff !important; font-family: 'Orbitron', sans-serif; font-size: 2.5rem !important; }
p, label { font-size: 1.4rem !important; color: #ffffff !important; font-family: 'Poppins', sans-serif; }

/* Action Buttons */
.stButton>button {
    background: linear-gradient(45deg, #ff00c1, #00d2ff);
    color: white !important;
    border-radius: 50px;
    padding: 15px 45px;
    font-weight: 900;
    width: 100%;
    border: none;
    box-shadow: 0 0 20px rgba(255, 0, 193, 0.4);
}
</style>
<div class="main-overlay"></div>
""", unsafe_allow_html=True)

# --- 3. CORE AI LOGIC FUNCTIONS ---

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
    # Unique frequency based on text letters
    freq = (sum([ord(c) for c in text]) % 400) + 150
    melody = 0.5 * np.sin(2 * np.pi * freq * t)
    return melody, sr

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h2 style='text-align:center; color:#00d2ff !important;'>TECHNOVA</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=120)
    st.write("---")
    
    # Sidebar with Specific Icons
    menu = st.radio(
        "SELECT MODULE",
        ["🏠 Dashboard", "🧠 Mood Spotify AI", "🎙️ Creative Studio", "♿ Hearing Assist"]
    )
    
    st.write("---")
    st.success("⚡ AI ENGINE : ONLINE")
    st.info("System Ver: 5.0")

# --- 5. HEADER BRANDING ---
st.markdown("<h1 class='hero-title'>TECHNOVA SOLUTION</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#92fe9d; font-weight:bold; letter-spacing:5px;'>SONICSENSE ULTRA PRO</p>", unsafe_allow_html=True)

# --- 6. MODULES ---

# --- 1. DASHBOARD ---
if "Dashboard" in menu:
    st.snow()
    col1, col2 = st.columns([1.6, 1])
    with col1:
        st.markdown("""<div class='glass-card'>
            <h2>Innovating Audio Through AI</h2>
            <p>Technova Solution bridges the gap between sound and AI. We create accessible audio tools for everyone, ensuring no one is left behind in the digital music era.</p>
            <ul>
                <li>Voice to Instrument Transformation</li>
                <li>AI Powered Mood Recommendation</li>
                <li>Inclusive Hearing Assistance</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.image("https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=500&h=500&fit=crop", use_container_width=True)

# --- 2. MOOD SPOTIFY AI (ERROR FIXED) ---
elif "Mood Spotify AI" in menu:
    st.markdown("<div class='glass-card'><h3>🧠 Mood-Based Smart Suggestions</h3></div>", unsafe_allow_html=True)
    
    # Verified Global Playlist IDs
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
            st.session_state.play_mood = mood_choice
            st.balloons()
            st.snow()

    with c2:
        if 'play_mood' in st.session_state:
            pid = mood_map[st.session_state.play_mood]
            # Embed structure that is stable and loads fast
            embed_url = f"https://open.spotify.com/embed/playlist/{pid}?utm_source=generator&theme=0"
            st.markdown(f"<h4 style='color:#1DB954;'>Vibe: {st.session_state.play_mood}</h4>", unsafe_allow_html=True)
            components.iframe(embed_url, height=450, scrolling=False)
        else:
            st.image("https://images.unsplash.com/photo-1493225255756-d9584f8606e9?q=80&w=800", use_container_width=True)

# --- 3. CREATIVE STUDIO (RECORD + UPLOAD + TEXT) ---
elif "Creative Studio" in menu:
    st.markdown("<div class='glass-card'><h3>🎙️ Creative AI Studio</h3></div>", unsafe_allow_html=True)
    
    t1, t2, t3 = st.tabs(["🎤 LIVE RECORD", "📤 UPLOAD FILE", "✍️ TEXT TO SONG"])
    
    with t1:
        st.write("Sing or speak to convert it into a melody.")
        live_voice = st.audio_input("Record now")
        if live_voice and st.button("✨ TRANSFORM LIVE VOICE"):
            y, sr = librosa.load(live_voice)
            out = voice_to_music(y, sr)
            st.audio(out, sample_rate=sr)
            st.balloons()

    with t2:
        st.write("Upload your voice file (MP3/WAV).")
        up_file = st.file_uploader("Choose file", type=["mp3", "wav"])
        if up_file and st.button("🚀 PROCESS UPLOAD"):
            y, sr = librosa.load(up_file)
            out = voice_to_music(y, sr)
            st.audio(out, sample_rate=sr)
            st.balloons()

    with t3:
        st.write("Type a word or sentence to generate an AI melody.")
        txt_input = st.text_input("Enter text (e.g., 'Technova Magic')")
        if txt_input and st.button("🎵 GENERATE MELODY"):
            mel, sr_mel = text_to_music(txt_input)
            st.audio(mel, sample_rate=sr_mel)
            st.balloons()

# --- 4. HEARING ASSIST (FREQUENCY LOGIC) ---
elif "Hearing Assist" in menu:
    st.markdown("<div class='glass-card'><h3>♿ Inclusive Hearing Assist</h3><p>Optimizing sound for the hearing impaired.</p></div>", unsafe_allow_html=True)
    
    h_file = st.file_uploader("Upload Audio for Pattern Optimization", type=["mp3", "wav"])
    if h_file:
        y, sr = librosa.load(h_file)
        
        # Frequency Shift logic:
        # Most hearing impaired individuals lose high-frequency hearing first.
        # Shifting the frequency down (Transpose) makes it "Bass-heavy" and feelable via vibrations.
        shift_val = st.slider("Adjust Sensitivity (Lower = More Bass/Vibration)", -12, 0, -8)
        
        if st.button("🔊 OPTIMIZE FOR VIBRATIONS"):
            st.snow()
            # Pitch shifting down to lower frequencies
            y_optimized = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift_val)
            
            # Boost Amplitude for bone conduction feel
            y_optimized = np.clip(y_optimized * 1.6, -1.0, 1.0)
            
            st.success("Audio Optimized! Use bone-conduction headphones or 'Earspots' to feel the patterns.")
            st.audio(y_optimized, sample_rate=sr)
            
            # Visual Pulse pattern
            st.line_chart(y_optimized[:15000])
