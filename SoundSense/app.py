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
    page_title="TECHNOVA SOLUTION | Pro Audio AI", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. PREMIUM NEON CSS (VIBRANT & VISIBLE) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&family=Rajdhani:wght@600;700&family=Inter:wght@500;800&display=swap');
    
    /* Background and Global */
    .stApp {
        background: linear-gradient(135deg, #050505 0%, #0a0a1a 100%);
        color: #ffffff;
    }

    /* TECHNOVA HEADER - BIG & BOLD */
    .technova-container {
        text-align: center;
        padding: 60px 20px;
        background: rgba(0, 0, 0, 0.6);
        border-radius: 50px;
        border: 3px solid #00d2ff;
        margin-bottom: 50px;
        box-shadow: 0 0 40px rgba(0, 210, 255, 0.4);
    }

    .company-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 6rem !important; /* Extremely Visible */
        font-weight: 900;
        background: linear-gradient(90deg, #ff00c1, #00d2ff, #92fe9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 15px;
        margin: 0;
    }

    .tagline {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2rem;
        color: #00d2ff;
        letter-spacing: 10px;
        text-transform: uppercase;
        font-weight: 700;
    }

    /* Content Visibility */
    h2, h3 { font-family: 'Orbitron', sans-serif; color: #ff00c1 !important; font-size: 2.8rem !important; }
    p, label { font-size: 1.5rem !important; font-weight: 600; color: #ffffff !important; }
    
    /* Neon Cards */
    .neon-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 30px;
        padding: 35px;
        border: 2px solid rgba(0, 210, 255, 0.3);
        margin-bottom: 30px;
        backdrop-filter: blur(10px);
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #ff00c1, #00d2ff);
        font-size: 1.8rem !important;
        font-family: 'Orbitron', sans-serif;
        padding: 20px;
        border-radius: 100px;
        box-shadow: 0 10px 30px rgba(255, 0, 193, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CORE LOGIC ---
def generate_music_from_audio(audio_data, sr):
    f0, _, _ = librosa.pyin(audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = np.nan_to_num(f0)
    hop_length = 512
    f0_stretched = np.repeat(f0, hop_length)[:len(audio_data)]
    phase = np.cumsum(2 * np.pi * f0_stretched / sr)
    music = 0.6 * np.sin(phase) + 0.3 * np.sin(2 * phase)
    return music / (np.max(np.abs(music)) + 1e-6)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='color:#00d2ff; font-family:Orbitron; font-size:2rem;'>TECHNOVA</h1>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1614149162883-504ce4d13909?q=80&w=200&auto=format&fit=crop", width=200)
    menu = ["🏠 Main Hub", "🎵 Spotify Mood AI", "🎤 Creative Studio", "🌈 Sensory Room", "♿ Accessibility Lab"]
    choice = st.sidebar.radio("NAVIGATE", menu)

# --- 5. TOP BRANDING ---
st.markdown("""
    <div class="technova-container">
        <h1 class="company-title">TECHNOVA</h1>
        <p class="tagline">SONICSENSE ULTRA PRO</p>
    </div>
    """, unsafe_allow_html=True)

# --- 6. MODULES ---

# --- HUB ---
if choice == "🏠 Main Hub":
    st.image("https://images.unsplash.com/photo-1514525253361-bee8718a7c7c?q=80&w=2064&auto=format&fit=crop", use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='neon-card'><h3>AI Vision</h3><p>Next-gen sound processing for everyone.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='neon-card'><h3>Accessibility</h3><p>Bridging the gap for hearing impaired users.</p></div>", unsafe_allow_html=True)

# --- SPOTIFY MOOD AI ---
elif choice == "🎵 Spotify Mood AI":
    st.title("Mood-Based Spotify Suggester")
    st.image("https://images.unsplash.com/photo-1493225255756-d9584f8606e9?q=80&w=1000&auto=format&fit=crop", width=800)
    
    u_mood = st.select_slider("Select your current mood:", 
                             options=["Depressed", "Sad", "Calm", "Happy", "Energetic", "Party Monster", "Devotional"])
    
    mood_urls = {
        "Depressed": "https://open.spotify.com/playlist/37i9dQZF1DX7qK8ma5wgG1",
        "Sad": "https://open.spotify.com/playlist/37i9dQZF1DX3YSRmBhyV9O",
        "Calm": "https://open.spotify.com/playlist/37i9dQZF1DX4sWSp4Status",
        "Happy": "https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0",
        "Energetic": "https://open.spotify.com/playlist/37i9dQZF1DX76W9SwwE6v4",
        "Party Monster": "https://open.spotify.com/playlist/37i9dQZF1DXaXB88o2P9G9",
        "Devotional": "https://open.spotify.com/playlist/37i9dQZF1DX0S69v9S94G0"
    }

    st.markdown(f"""
        <div class='neon-card'>
            <h2>Vibe: {u_mood}</h2>
            <p>Technova AI recommends this playlist for your soul:</p>
            <a href='{mood_urls[u_mood]}' target='_blank'>
                <button style='background:#1DB954; color:white; border:none; padding:20px 40px; border-radius:50px; font-weight:bold; width:100%; cursor:pointer; font-size:1.5rem;'>
                    🎧 OPEN {u_mood.upper()} PLAYLIST ON SPOTIFY
                </button>
            </a>
        </div>
    """, unsafe_allow_html=True)

# --- CREATIVE STUDIO ---
elif choice == "🎤 Creative Studio":
    st.title("Technova Live Studio")
    
    t1, t2 = st.tabs(["🎙️ LIVE VOICE", "✍️ TEXT TO SOUND"])
    
    with t1:
        v_input = st.audio_input("Record your voice / song:")
        if v_input:
            y, sr = librosa.load(v_input)
            if st.button("✨ TRANSFORM TO INSTRUMENT"):
                music = generate_music_from_audio(y, sr)
                st.audio(music, sample_rate=sr)
                st.image("https://images.unsplash.com/photo-1614613535308-eb5fbd3d2c17?q=80&w=300&h=300&fit=crop", caption="Technova AI Generated Album Art")

    with t2:
        text = st.text_input("Enter text to create a melody:")
        if text:
            st.info(f"Generating unique frequency for '{text}'...")
            st.audio(np.random.uniform(-1,1, 44100), sample_rate=44100) # Placeholder

# --- SENSORY ROOM ---
elif choice == "🌈 Sensory Room":
    st.title("Visual Sensory Room")
    st.image("https://images.unsplash.com/photo-1550684848-fac1c5b4e853?q=80&w=1000", width=800)
    up = st.file_uploader("Upload Audio", type=['mp3','wav'])
    if up:
        y, sr = librosa.load(up)
        st.line_chart(y[:5000], color="#00d2ff")

# --- ACCESSIBILITY LAB ---
elif choice == "♿ Accessibility Lab":
    st.title("Inclusive Hearing Assist")
    st.markdown("<div class='neon-card'><h3>Haptic Pattern Simulator</h3><p>See the pattern of the music in vibrations.</p></div>", unsafe_allow_html=True)
    
    test_audio = st.file_uploader("Upload sound to see 'Haptic Pulse'", type=['mp3','wav'])
    if test_audio:
        y, sr = librosa.load(test_audio)
        rms = librosa.feature.rms(y=y)[0]
        st.subheader("Vibration Intensity (Pulse)")
        # Display as a moving pulse bar
        st.progress(min(int(np.max(rms)*500), 100))
        st.write("Pattern detected. Use Earspots for physical haptic feedback.")
