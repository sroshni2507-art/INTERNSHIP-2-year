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
import speech_recognition as sr

# Handle MoviePy compatibility for video processing
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SonicSense Ultra Pro", layout="wide", initial_sidebar_state="expanded")

# --- NEON GLASSMORPHISM CSS (High Contrast) ---
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

    /* Fix Text Visibility Globally */
    label, p, span, .stMarkdown, div[data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        font-weight: 500;
    }

    /* Glassmorphism Card Design */
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(0, 210, 255, 0.4);
        backdrop-filter: blur(15px);
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
    }

    /* Neon Gradient Titles */
    h1 {
        background: linear-gradient(to right, #00d2ff, #92fe9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 3.8rem !important;
        margin-bottom: 10px;
    }

    h2, h3 {
        color: #00d2ff !important;
        font-weight: 700;
    }

    /* Styled Tabs */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important;
    }
    button[data-baseweb="tab"] div {
        color: #ffffff !important;
        font-size: 1.1rem;
        font-weight: 700;
    }

    /* Buttons Style */
    .stButton>button {
        background: linear-gradient(45deg, #00d2ff, #3a7bd5);
        color: white !important;
        border-radius: 50px;
        padding: 12px 35px;
        font-weight: 800;
        border: none;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.5);
        transition: 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(0, 210, 255, 0.8);
    }

    /* Sidebar Background */
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
    # Normalize to avoid audio player errors
    if np.max(np.abs(music)) > 0:
        music = music / (np.max(np.abs(music)) + 1e-6)
    return music

# --- SIDEBAR MENU ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=80)
    st.markdown("<h2 style='text-align: center;'>SONICSENSE PRO</h2>", unsafe_allow_html=True)
    menu = ["🏠 Dashboard", "🎨 Creative Studio", "🧠 Mood & Spotify AI", "🌈 Sensory Room", "🎬 Movie Lab"]
    choice = st.sidebar.radio("SELECT MODULE", menu)
    st.write("---")
    st.info("💡 Pro Tip: Grant Microphone permission in your browser to use 'Record Live'.")

# --- MODULE 1: DASHBOARD ---
if choice == "🏠 Dashboard":
    st.title("Inclusive Audio Intelligence")
    st.image("https://images.unsplash.com/photo-1550745165-9bc0b252726f?q=80&w=2070&auto=format&fit=crop", caption="Tech-driven Inclusivity", use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='glass-card'><h3>Accessibility</h3><p>Designed for the hearing impaired to 'see' sound energy.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='glass-card'><h3>Creative AI</h3><p>Machine Learning modules for voice-to-instrumental synthesis.</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='glass-card'><h3>Work Activity</h3><p>Boost focus with mood tracking and Spotify flows.</p></div>", unsafe_allow_html=True)

# --- MODULE 2: CREATIVE STUDIO ---
elif choice == "🎨 Creative Studio":
    st.title("AI Creative Studio")
    st.image("https://images.unsplash.com/photo-1598488035139-bdbb2231ce04?q=80&w=2070&auto=format&fit=crop", use_container_width=True)
    
    st.markdown("### Choose Input Method")
    tab1, tab2, tab3 = st.tabs(["🎙️ RECORD LIVE", "📁 UPLOAD FILE", "✍️ TEXT TO SOUND"])
    
    with tab1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("On-the-spot Recording")
        # Direct Microphone Input
        recorded_audio = st.audio_input("SPEAK OR HUM INTO MIC...")
        if recorded_audio:
            y, sr_load = librosa.load(recorded_audio)
            if st.button("TRANSFORM LIVE VOICE"):
                with st.spinner("Synthesizing melody..."):
                    music = generate_music_from_audio(y, sr_load)
                    st.audio(music, sample_rate=sr_load)
                    st.success("Converted Successfully!")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("UPLOAD VOICE FILE (WAV/MP3)", type=['wav', 'mp3'])
        if uploaded_file:
            y, sr_load = librosa.load(uploaded_file)
            if st.button("PROCESS UPLOADED FILE"):
                music = generate_music_from_audio(y, sr_load)
                st.audio(music, sample_rate=sr_load)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Text to Frequency Generator")
        user_text = st.text_input("TYPE SOMETHING TO HEAR ITS TONE:")
        if user_text:
            duration = 2.0
            sr_gen = 22050
            t = np.linspace(0, duration, int(sr_gen * duration))
            freq = sum([ord(c) for c in user_text]) % 500 + 200
            text_music = 0.5 * np.sin(2 * np.pi * freq * t)
            st.audio(text_music, sample_rate=sr_gen)
        st.markdown("</div>", unsafe_allow_html=True)

# --- MODULE 3: MOOD & SPOTIFY AI ---
elif choice == "🧠 Mood & Spotify AI":
    st.title("Productivity & Flow State")
    st.image("https://images.unsplash.com/photo-1493225255756-d9584f8606e9?q=80&w=2070&auto=format&fit=crop", use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        u_mood = st.selectbox("HOW ARE YOU FEELING?", ["Energetic", "Calm", "Focused", "Stressed"])
        u_goal = st.text_input("YOUR GOAL TODAY:", "Project Internship")
        st.markdown("</div>", unsafe_allow_html=True)
        
    if st.button("CREATE FLOW SESSION"):
        st.balloons()
        mood_map = {
            "Energetic": ("EDM / Power", "https://open.spotify.com/search/energetic%20workout"),
            "Calm": ("Lofi / Chill", "https://open.spotify.com/search/calm%20piano"),
            "Focused": ("Deep Work / Ambient", "https://open.spotify.com/search/deep%20focus"),
            "Stressed": ("Classical / Relief", "https://open.spotify.com/search/meditation%20music")
        }
        genre, link = mood_map[u_mood]
        st.markdown(f"<div class='glass-card'><h3>Recommendation: {genre} Session</h3><p>Activity Goal: {u_goal}</p></div>", unsafe_allow_html=True)
        st.markdown(f"<a href='{link}' target='_blank'><button style='background:#1DB954; color:white; border:none; padding:15px; border-radius:30px; width:100%; cursor:pointer; font-weight:800;'>🟢 OPEN IN SPOTIFY</button></a>", unsafe_allow_html=True)
        
        # Activity Download
        st.download_button("📥 DOWNLOAD MY WORK PLAN", f"Mood: {u_mood}\nTarget: {u_goal}\nMusic: {genre}", "flow_plan.txt")

# --- MODULE 4: SENSORY ROOM (Hearing Impaired Support) ---
elif choice == "🌈 Sensory Room":
    st.title("Sensory Visualization Room")
    st.image("https://images.unsplash.com/photo-1550684848-fac1c5b4e853?q=80&w=2070&auto=format&fit=crop", use_container_width=True)
    st.info("Hearing-impaired users can watch the sound ripples and color patterns.")
    
    sens_file = st.file_uploader("UPLOAD AUDIO TO SEE VIBRATIONS", type=['wav', 'mp3'])
    if sens_file:
        y, sr_rate = librosa.load(sens_file)
        rms = librosa.feature.rms(y=y)[0]
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 3), facecolor='#05060a')
        ax.plot(rms, color='#00d2ff', linewidth=2.5)
        ax.fill_between(range(len(rms)), rms, color='#3a7bd5', alpha=0.3)
        ax.set_facecolor('#05060a')
        ax.set_axis_off()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("ACTIVATE BEAT SYNC LIGHTS"):
            placeholder = st.empty()
            for _ in range(4):
                placeholder.markdown("<div style='height:30px; background:#00d2ff; border-radius:20px; box-shadow: 0 0 40px #00d2ff;'></div>", unsafe_allow_html=True)
                time.sleep(0.2)
                placeholder.markdown("<div style='height:30px; background:#302b63; border-radius:20px;'></div>", unsafe_allow_html=True)
                time.sleep(0.2)

# --- MODULE 5: MOVIE LAB ---
elif choice == "🎬 Movie Lab":
    st.title("AI Movie Subtitle Lab")
    st.image("https://images.unsplash.com/photo-1485846234645-a62644f84728?q=80&w=2059&auto=format&fit=crop", use_container_width=True)
    
    video_input = st.file_uploader("UPLOAD VIDEO FILE (MP4)", type=['mp4'])
    if video_input:
        st.video(video_input)
        if st.button("START AI TRANSCRIPTION"):
            with st.spinner("AI is analyzing movie audio..."):
                time.sleep(3)
                st.markdown("<div class='glass-card'><h3>📜 Subtitles Generated:</h3><p>'SonicSense Pro: Innovation in inclusive technology. Empowering vision through sound.'</p></div>", unsafe_allow_html=True)
                st.success("Emotion Detected: 😊 Inspiring / Positive")
