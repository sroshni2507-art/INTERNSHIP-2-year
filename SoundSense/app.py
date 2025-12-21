import streamlit as st
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import io
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TECHNOVA SOLUTION | Ultimate SonicSense", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. ULTRA-CREATIVE NEON CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&family=Rajdhani:wght@500;700&display=swap');
    
    /* Background & Scrollbar */
    .stApp {
        background: radial-gradient(circle at center, #111122 0%, #050505 100%);
        color: #ffffff;
    }

    /* TECHNOVA PREMIUM HEADER */
    .technova-container {
        text-align: center;
        padding: 50px 20px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 40px;
        border: 2px solid #00d2ff;
        margin-bottom: 40px;
        box-shadow: 0 0 30px rgba(0, 210, 255, 0.3), inset 0 0 20px rgba(0, 210, 255, 0.1);
        backdrop-filter: blur(15px);
    }

    .company-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 5.5rem !important;
        font-weight: 900;
        letter-spacing: 12px;
        background: linear-gradient(to right, #00d2ff, #bc13fe, #92fe9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(0, 210, 255, 0.5);
        margin: 0;
    }

    .tagline {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.8rem;
        color: #92fe9d;
        letter-spacing: 8px;
        text-transform: uppercase;
        font-weight: 700;
        margin-top: 10px;
    }

    /* Visibility and Font Sizes */
    h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #00d2ff !important;
        font-size: 2.5rem !important;
        margin-bottom: 20px;
    }
    
    p, label, .stMarkdown {
        font-size: 1.3rem !important;
        font-weight: 500;
        color: #e0e0e0;
    }

    /* Colorful Glass Cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.07);
        border-radius: 25px;
        padding: 30px;
        border-left: 10px solid #bc13fe;
        margin-bottom: 25px;
        transition: 0.4s;
    }
    .feature-card:hover {
        background: rgba(255, 255, 255, 0.12);
        transform: scale(1.02);
        border-left: 10px solid #00d2ff;
    }

    /* Neon Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #bc13fe, #00d2ff);
        color: white !important;
        border-radius: 50px;
        padding: 15px 45px;
        font-size: 1.5rem !important;
        font-family: 'Orbitron', sans-serif;
        border: none;
        box-shadow: 0 0 20px rgba(188, 19, 254, 0.5);
        cursor: pointer;
        width: 100%;
    }
    .stButton>button:hover {
        box-shadow: 0 0 40px rgba(0, 210, 255, 0.8);
        transform: translateY(-3px);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CORE LOGIC FUNCTIONS ---

def voice_to_music(y, sr):
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = np.nan_to_num(f0)
    hop_length = 512
    f0_stretched = np.repeat(f0, hop_length)[:len(y)]
    phase = np.cumsum(2 * np.pi * f0_stretched / sr)
    # Creating a rich synthesizer sound
    music = 0.6 * np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
    music[f0_stretched == 0] = 0
    return music / (np.max(np.abs(music)) + 1e-6)

def text_to_melody(text):
    sr = 22050
    duration = max(len(text) * 0.2, 3.0)
    t = np.linspace(0, duration, int(sr * duration))
    # Generate unique frequency based on text characters
    base_freq = (sum([ord(c) for c in text]) % 400) + 150
    melody = 0.5 * np.sin(2 * np.pi * base_freq * t) * np.exp(-t/duration)
    return melody, sr

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='color:#bc13fe; font-family:Orbitron;'>TECHNOVA</h1>", unsafe_allow_html=True)
    menu = ["🚀 Home Hub", "🎤 Creative AI Studio", "👂 Hearing Assist", "🌈 Sensory Pulse"]
    choice = st.sidebar.selectbox("Navigate Features", menu)
    st.write("---")
    st.success("AI Core: Online")

# --- 5. TOP HEADER ---
st.markdown("""
    <div class="technova-container">
        <h1 class="company-title">TECHNOVA SOLUTION</h1>
        <p class="tagline">The Future of Sensory Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

# --- 6. MODULES ---

if choice == "🚀 Home Hub":
    st.markdown("## Global Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class='feature-card'>
            <h3>Live Studio</h3>
            <p>Convert your real-time voice into high-quality digital music.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='feature-card'>
            <h3>Text-to-Tune</h3>
            <p>Type your thoughts and let Technova AI generate a unique melody.</p>
        </div>""", unsafe_allow_html=True)

elif choice == "🎤 Creative AI Studio":
    st.markdown("## Creative AI Studio")
    
    tab1, tab2, tab3 = st.tabs(["🎙️ LIVE VOICE TO MUSIC", "📤 UPLOAD & CONVERT", "✍️ TEXT TO MELODY"])
    
    with tab1:
        st.write("### On-the-spot Voice Conversion")
        voice_rec = st.audio_input("Speak or Sing now to create music:")
        if voice_rec:
            y, sr = librosa.load(voice_rec)
            if st.button("✨ CONVERT LIVE VOICE"):
                with st.spinner("Technova AI generating your unique sound..."):
                    music = voice_to_music(y, sr)
                    st.audio(music, sample_rate=sr)
                    st.balloons()

    with tab2:
        st.write("### Upload and Re-imagine")
        up_file = st.file_uploader("Upload any voice or song file", type=['wav', 'mp3'])
        if up_file:
            y, sr = librosa.load(up_file)
            if st.button("🚀 PROCESS UPLOAD"):
                music = voice_to_music(y, sr)
                st.audio(music, sample_rate=sr)

    with tab3:
        st.write("### AI Text-to-Sound Generator")
        user_text = st.text_input("Enter a word or sentence (e.g., 'Technova Magic'):")
        if user_text:
            if st.button("🎵 GENERATE MELODY"):
                melody, sr_gen = text_to_melody(user_text)
                st.audio(melody, sample_rate=sr_gen)
                st.info(f"Generated a unique melody for: '{user_text}'")

elif choice == "👂 Hearing Assist":
    st.markdown("## Inclusive Hearing Lab")
    st.write("Optimizing sound for Hearing Impaired & Bone Conduction users.")
    up_h = st.file_uploader("Upload Audio for Frequency Shift", type=['wav', 'mp3'], key="hear")
    if up_h:
        y, sr = librosa.load(up_h)
        shift = st.slider("Select Pitch Shifting (Lower = Easier to feel)", -15, 0, -7)
        if st.button("🔊 OPTIMIZE FOR VIBRATION"):
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift)
            y_final = np.clip(y_shifted * 1.8, -1.0, 1.0)
            st.audio(y_final, sample_rate=sr)
            st.success("Sound optimized for 'Earspots' vibration mode.")

elif choice == "🌈 Sensory Pulse":
    st.markdown("## Real-time Sensory Visualization")
    v_file = st.file_uploader("Upload to see the pulse", type=['wav', 'mp3'])
    if v_file:
        y, sr = librosa.load(v_file)
        rms = librosa.feature.rms(y=y)[0]
        fig, ax = plt.subplots(figsize=(12, 4), facecolor='#050505')
        ax.fill_between(range(len(rms)), rms, color='#00d2ff', alpha=0.7)
        ax.set_axis_off()
        st.pyplot(fig)
