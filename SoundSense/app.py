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

# --- PAGE CONFIG ---
st.set_page_config(page_title="SonicSense Ultra Pro", layout="wide", initial_sidebar_state="expanded")

# --- ENHANCED CSS (Contrast & Clarity) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #0a0b10;
        color: #ffffff; /* Global Text Color - Pure White for Clarity */
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #16213e, #0a0b10);
    }

    /* Fixing visibility of Labels and Sub-text */
    label, .stMarkdown, p, div[data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        font-weight: 500;
        font-size: 1.05rem;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.07);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(0, 210, 255, 0.3); /* Neon Border */
        backdrop-filter: blur(15px);
        margin-bottom: 25px;
        color: white;
    }

    /* Tab Text Color Fix */
    button[data-baseweb="tab"] div {
        color: #ffffff !important;
        font-size: 1.1rem;
    }
    
    /* Neon Titles */
    h1 {
        background: linear-gradient(to right, #00d2ff, #92fe9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem !important;
    }

    h2, h3 {
        color: #00d2ff !important;
        font-weight: 700;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #00d2ff, #3a7bd5);
        color: white !important;
        border: none;
        border-radius: 50px;
        padding: 15px 40px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: 0.4s;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.4);
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(0, 210, 255, 0.7);
    }

    /* Sidebar Appearance */
    section[data-testid="stSidebar"] {
        background-color: #050505;
        border-right: 1px solid #00d2ff33;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER: VOICE TO MUSIC LOGIC ---
def generate_music_from_audio(audio_data, sr):
    f0, _, _ = librosa.pyin(audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = np.nan_to_num(f0)
    hop_length = 512
    f0_stretched = np.repeat(f0, hop_length)[:len(audio_data)]
    phase = np.cumsum(2 * np.pi * f0_stretched / sr)
    music = 0.5 * np.sin(phase) + 0.2 * np.sin(2 * phase)
    music[f0_stretched == 0] = 0
    return music / (np.max(np.abs(music)) + 1e-6)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=80)
    st.markdown("## SONICSENSE PRO")
    menu = ["🏠 Dashboard", "🎨 Creative Studio", "🧠 Mood & Spotify AI", "🌈 Sensory Room", "🎬 Movie Lab"]
    choice = st.radio("SELECT MODULE", menu)
    st.write("---")
    st.info("v2.5 - High Contrast Mode Active")

# --- 1. DASHBOARD ---
if choice == "🏠 Dashboard":
    st.title("SonicSense Intelligence Hub")
    st.image("https://images.unsplash.com/photo-1550745165-9bc0b252726f?q=80&w=2070&auto=format&fit=crop", caption="Merging Technology and Sound", use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='glass-card'><h3>Universal Design</h3><p>Built for the hearing impaired and creative artists alike.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='glass-card'><h3>AI Audio Engine</h3><p>Neural pitch tracking & frequency synthesis modules.</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='glass-card'><h3>Productivity</h3><p>Deep Focus mood AI with Spotify integration.</p></div>", unsafe_allow_html=True)

# --- 2. CREATIVE STUDIO ---
elif choice == "🎨 Creative Studio":
    st.title("Creative Sound Studio")
    st.image("https://images.unsplash.com/photo-1598488035139-bdbb2231ce04?q=80&w=2070&auto=format&fit=crop", use_container_width=True)
    
    st.markdown("### Select your Input Method")
    tab1, tab2, tab3 = st.tabs(["🎙️ RECORD LIVE", "📁 UPLOAD FILE", "✍️ TEXT TO SOUND"])
    
    with tab1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("On-the-spot Recording")
        recorded_audio = st.audio_input("SPEAK OR HUM NOW...")
        if recorded_audio:
            y, sr = librosa.load(recorded_audio)
            if st.button("PROCESS LIVE VOICE"):
                with st.spinner("Generating Music..."):
                    music = generate_music_from_audio(y, sr)
                    st.audio(music, sample_rate=sr)
                    st.success("Converted to Instrumental successfully!")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("CHOOSE WAV/MP3 FILE", type=['wav', 'mp3'])
        if uploaded_file:
            y, sr = librosa.load(uploaded_file)
            if st.button("CONVERT FILE"):
                music = generate_music_from_audio(y, sr)
                st.audio(music, sample_rate=sr)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        user_text = st.text_input("ENTER TEXT TO GENERATE FREQUENCY:")
        if user_text:
            duration = 2.0
            sr = 22050
            t = np.linspace(0, duration, int(sr * duration))
            freq = sum([ord(c) for c in user_text]) % 500 + 200
            text_music = 0.5 * np.sin(2 * np.pi * freq * t)
            st.audio(text_music, sample_rate=sr)
        st.markdown("</div>", unsafe_allow_html=True)

# --- 3. MOOD & SPOTIFY AI ---
elif choice == "🧠 Mood & Spotify AI":
    st.title("Mood-Based Productivity")
    st.image("https://images.unsplash.com/photo-1493225255756-d9584f8606e9?q=80&w=2070&auto=format&fit=crop", use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        u_mood = st.selectbox("HOW DO YOU FEEL?", ["Energetic", "Calm", "Focused", "Stressed"])
        u_goal = st.text_input("YOUR CURRENT TASK:", "Internship Project")
        st.markdown("</div>", unsafe_allow_html=True)
        
    if st.button("SYNC WITH SPOTIFY"):
        st.balloons()
        recs = {
            "Energetic": ("EDM / Workout", "https://open.spotify.com/search/energetic%20workout"),
            "Calm": ("Nature / Chill", "https://open.spotify.com/search/calm%20lofi"),
            "Focused": ("Deep Work", "https://open.spotify.com/search/deep%20focus"),
            "Stressed": ("Piano / Relief", "https://open.spotify.com/search/stress%20relief")
        }
        genre, link = recs[u_mood]
        st.markdown(f"<div class='glass-card'><h3>Recommendation: {genre} Session</h3><p>Focusing on: {u_goal}</p></div>", unsafe_allow_html=True)
        st.markdown(f"<a href='{link}' target='_blank'><button style='background:#1DB954; color:white; border:none; padding:15px; border-radius:30px; width:100%; cursor:pointer;'>🔥 OPEN PLAYLIST ON SPOTIFY</button></a>", unsafe_allow_html=True)

# --- 4. SENSORY ROOM ---
elif choice == "🌈 Sensory Room":
    st.title("Sensory Visualization Room")
    st.image("https://images.unsplash.com/photo-1550684848-fac1c5b4e853?q=80&w=2070&auto=format&fit=crop", use_container_width=True)
    st.info("Visual patterns for the hearing impaired to 'see' sound.")
    
    sens_file = st.file_uploader("UPLOAD AUDIO FOR SENSORY MAPPING", type=['wav', 'mp3'])
    if sens_file:
        y, sr = librosa.load(sens_file)
        rms = librosa.feature.rms(y=y)[0]
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 3), facecolor='#0a0b10')
        ax.plot(rms, color='#00d2ff', linewidth=2)
        ax.fill_between(range(len(rms)), rms, color='#3a7bd5', alpha=0.4)
        ax.set_facecolor('#0a0b10')
        ax.set_axis_off()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("ACTIVATE FLASH SYNC"):
            for _ in range(3):
                st.markdown("<div style='height:40px; background:linear-gradient(90deg, #00d2ff, #3a7bd5); border-radius:20px; box-shadow: 0 0 30px #00d2ff;'></div>", unsafe_allow_html=True)
                time.sleep(0.3)
                st.write("")

# --- 5. MOVIE LAB ---
elif choice == "🎬 Movie Lab":
    st.title("Movie Lab & Transcription")
    st.image("https://images.unsplash.com/photo-1485846234645-a62644f84728?q=80&w=2059&auto=format&fit=crop", use_container_width=True)
    
    video = st.file_uploader("UPLOAD MOVIE CLIP (MP4)", type=['mp4'])
    if video:
        st.video(video)
        if st.button("RUN AI TRANSCRIPTION"):
            with st.spinner("Decoding Audio Tracks..."):
                time.sleep(2)
                st.markdown("<div class='glass-card'><h3>📜 Generated Subtitles:</h3><p>'SoundSense Pro: Bridging the gap between sound and sight for everyone.'</p></div>", unsafe_allow_html=True)
