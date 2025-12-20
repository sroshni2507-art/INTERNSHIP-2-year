import streamlit as st
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
import io
import time
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SonicSense Ultra Pro", layout="wide", initial_sidebar_state="expanded")

# --- NEON GLASSMORPHISM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #05060a;
        color: #ffffff !important;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1a1b2e, #05060a);
    }

    /* Clear Text Visibility */
    label, p, span, .stMarkdown, div[data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        font-weight: 600;
        text-shadow: 1px 1px 3px black;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(0, 210, 255, 0.5);
        backdrop-filter: blur(15px);
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
    }

    h1 {
        background: linear-gradient(to right, #00d2ff, #92fe9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 3.5rem !important;
    }

    h2, h3 {
        color: #00d2ff !important;
        font-weight: 700;
    }

    .stButton>button {
        background: linear-gradient(45deg, #00d2ff, #3a7bd5);
        color: white !important;
        border-radius: 50px;
        padding: 12px 40px;
        font-weight: 800;
        border: none;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.5);
    }

    section[data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #00d2ff33;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CORE LOGIC: VOICE TO INSTRUMENT ---
def generate_music_from_audio(audio_data, sr):
    f0, _, _ = librosa.pyin(audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = np.nan_to_num(f0)
    hop_length = 512
    f0_stretched = np.repeat(f0, hop_length)[:len(audio_data)]
    phase = np.cumsum(2 * np.pi * f0_stretched / sr)
    music = 0.5 * np.sin(phase) + 0.2 * np.sin(2 * phase)
    music[f0_stretched == 0] = 0
    if np.max(np.abs(music)) > 0:
        music = music / (np.max(np.abs(music)) + 1e-6)
    return music

# --- SIDEBAR MENU ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=80)
    st.markdown("<h2 style='text-align: center; color:white !important;'>SONICSENSE PRO</h2>", unsafe_allow_html=True)
    menu = ["🏠 Dashboard", "🎨 Creative Studio", "🧠 Mood & Spotify AI", "🌈 Sensory Room", "🎬 Movie Lab"]
    choice = st.sidebar.radio("SELECT MODULE", menu)
    st.write("---")
    st.success("Microphone Status: Ready")

# --- 1. DASHBOARD ---
if choice == "🏠 Dashboard":
    st.title("Inclusive Audio AI Hub")
    st.image("https://images.unsplash.com/photo-1550745165-9bc0b252726f?q=80&w=2070&auto=format&fit=crop", use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='glass-card'><h3>Accessibility</h3><p>Sound vibrations for the hearing impaired.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='glass-card'><h3>AI Studio</h3><p>Voice-to-Instrument synthesis.</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='glass-card'><h3>Productivity</h3><p>Spotify sessions for Flow State.</p></div>", unsafe_allow_html=True)

# --- 2. CREATIVE STUDIO ---
elif choice == "🎨 Creative Studio":
    st.title("Creative Sound Studio")
    st.image("https://images.unsplash.com/photo-1598488035139-bdbb2231ce04?q=80&w=2070&auto=format&fit=crop", use_container_width=True)
    
    tab1, tab2, tab3 = st.tabs(["🎙️ RECORD LIVE", "📁 UPLOAD FILE", "✍️ TEXT TO SOUND"])
    
    with tab1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Live Recording")
        recorded_audio = st.audio_input("Click to record your voice")
        if recorded_audio:
            y, sr_load = librosa.load(recorded_audio)
            if st.button("✨ CONVERT VOICE TO MUSIC"):
                music = generate_music_from_audio(y, sr_load)
                st.audio(music, sample_rate=sr_load)
                # Download
                buffer = io.BytesIO()
                sf.write(buffer, music, sr_load, format='WAV')
                st.download_button("💾 Download AI Music", buffer, "ai_music.wav")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Audio", type=['wav', 'mp3'])
        if uploaded_file:
            y, sr_load = librosa.load(uploaded_file)
            if st.button("PROCESS UPLOADED FILE"):
                music = generate_music_from_audio(y, sr_load)
                st.audio(music, sample_rate=sr_load)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Text to Sound Generator")
        user_text = st.text_input("Type a message to generate a melody:")
        if user_text:
            # FIXED: Duration scales with text length
            duration = min(max(len(user_text) * 0.1, 3.0), 10.0) 
            sr_gen = 22050
            t = np.linspace(0, duration, int(sr_gen * duration))
            
            # Create a complex melodic tone based on text
            freq_base = sum([ord(c) for c in user_text]) % 300 + 200
            text_music = 0.5 * np.sin(2 * np.pi * freq_base * t) + 0.2 * np.sin(4 * np.pi * freq_base * t)
            
            st.write(f"Generated Melody Duration: {round(duration, 2)} seconds")
            st.audio(text_music, sample_rate=sr_gen)
            
            # Download
            buffer = io.BytesIO()
            sf.write(buffer, text_music, sr_gen, format='WAV')
            st.download_button("💾 Download Text-Tone", buffer, "text_melody.wav")
        st.markdown("</div>", unsafe_allow_html=True)

# --- 3. MOOD & SPOTIFY AI ---
elif choice == "🧠 Mood & Spotify AI":
    st.title("Focus & Mood Flow")
    st.image("https://images.unsplash.com/photo-1493225255756-d9584f8606e9?q=80&w=2070&auto=format&fit=crop", use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        u_mood = st.selectbox("Mood", ["Energetic", "Calm", "Focused", "Stressed"])
        u_goal = st.text_input("Goal:", "Study Session")
        st.markdown("</div>", unsafe_allow_html=True)
        
    if st.button("GENERATE SPOTIFY FLOW"):
        st.balloons()
        mood_map = {"Energetic": "Workout", "Calm": "Lofi Chill", "Focused": "Focus Piano", "Stressed": "Zen"}
        st.markdown(f"<div class='glass-card'><h3>Flow: {mood_map[u_mood]}</h3><p>Task: {u_goal}</p></div>", unsafe_allow_html=True)
        st.markdown(f"<a href='https://open.spotify.com' target='_blank'><button style='background:#1DB954; color:white; border:none; padding:15px; border-radius:30px; width:100%; cursor:pointer;'>🎧 OPEN SPOTIFY</button></a>", unsafe_allow_html=True)

# --- 4. SENSORY ROOM ---
elif choice == "🌈 Sensory Room":
    st.title("Sensory Visualization Room")
    st.image("https://images.unsplash.com/photo-1550684848-fac1c5b4e853?q=80&w=2070&auto=format&fit=crop", use_container_width=True)
    
    sens_file = st.file_uploader("Upload sound to 'See' vibrations", type=['wav', 'mp3'])
    if sens_file:
        y, sr_rate = librosa.load(sens_file)
        rms = librosa.feature.rms(y=y)[0]
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 3), facecolor='#05060a')
        ax.plot(rms, color='#00d2ff', linewidth=3)
        ax.fill_between(range(len(rms)), rms, color='#3a7bd5', alpha=0.3)
        ax.set_axis_off()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

# --- 5. MOVIE LAB ---
elif choice == "🎬 Movie Lab":
    st.title("Movie AI Subtitles")
    st.image("https://images.unsplash.com/photo-1485846234645-a62644f84728?q=80&w=2059&auto=format&fit=crop", use_container_width=True)
    video_input = st.file_uploader("Upload Movie (MP4)", type=['mp4'])
    if video_input:
        st.video(video_input)
        if st.button("GENERATE AI SUBTITLES"):
            st.markdown("<div class='glass-card'><h3>📜 AI Subtitles:</h3><p>'Innovation for everyone: Bridging sound and vision.'</p></div>", unsafe_allow_html=True)
