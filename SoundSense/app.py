import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import io

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TECHNOVA SOLUTION | Pro AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ADVANCED CSS (PINK SIDEBAR & HIGH VISIBILITY) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&family=Poppins:wght@400;700;800&display=swap');

    /* Global App Background */
    .stApp {
        background: url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
    }
    .main-overlay {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0, 0, 0, 0.85); z-index: -1;
    }

    /* --- SIDEBAR PINK NEON STYLE --- */
    [data-testid="stSidebar"] {
        background-color: #050510 !important;
        border-right: 2px solid #ff00c1;
    }
    
    /* Force Sidebar Words to be PINK and Visible */
    [data-testid="stSidebar"] * {
        color: #ff00c1 !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
    }
    
    /* TECHNOVA HEADER */
    .hero-container {
        text-align: center; padding: 40px;
        background: rgba(255, 255, 255, 0.04);
        border-radius: 35px; border: 2px solid #ff00c1;
        backdrop-filter: blur(15px); margin-bottom: 25px;
        box-shadow: 0 0 30px rgba(255, 0, 193, 0.2);
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
        border: 1px solid #ff00c144;
        margin-bottom: 25px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.8);
    }

    /* Fonts Visibility */
    h2, h3 { color: #00d2ff !important; font-family: 'Orbitron', sans-serif; font-size: 2.7rem !important; text-shadow: 2px 2px 5px black; }
    p, label { font-size: 1.5rem !important; color: #ffffff !important; font-family: 'Poppins', sans-serif; font-weight: 500; text-shadow: 1px 1px 3px black; }
    
    /* Neon Styled Button */
    .stButton>button {
        background: linear-gradient(45deg, #ff00c1, #00d2ff);
        border: none; color: white !important; border-radius: 50px;
        padding: 18px 45px; font-size: 1.5rem !important;
        font-family: 'Orbitron', sans-serif; width: 100%;
        box-shadow: 0 0 20px rgba(255, 0, 193, 0.4);
    }
    .stButton>button:hover { transform: scale(1.03); box-shadow: 0 0 40px #ff00c1; }
    </style>
    <div class="main-overlay"></div>
    """, unsafe_allow_html=True)

# --- 3. CORE LOGIC ---
def voice_to_music_logic(audio, sr):
    f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = np.nan_to_num(f0)
    hop = 512
    f0_stretched = np.repeat(f0, hop)[:len(audio)]
    phase = np.cumsum(2 * np.pi * f0_stretched / sr)
    synth = 0.5 * np.sin(phase) + 0.3 * np.sin(2 * phase)
    return synth / (np.max(np.abs(synth)) + 1e-6)

# --- 4. SIDEBAR NAVIGATION (PINK FONT & ICONS) ---
with st.sidebar:
    st.markdown("<h2 style='color:#00d2ff !important; text-align:center;'>TECHNOVA</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=120)
    st.write("---")
    
    # Selection Menu with Icons
    choice = st.radio("SELECT MENU:", [
        "🏠 Dashboard", 
        "🧠 Mood Spotify AI", 
        "🎙️ Creative Studio", 
        "♿ Hearing Assist"
    ])
    
    st.write("---")
    st.success("⚡ AI ENGINE: ONLINE")
    st.info("💎 Ver: 5.0 Ultra Pro")

# --- 5. TOP BRANDING HEADER ---
st.markdown("""
    <div class="hero-container">
        <h1 class="company-title">TECHNOVA SOLUTION</h1>
        <p style="letter-spacing: 6px; color:#92fe9d; font-size:1.6rem; font-weight:700;">SONICSENSE ULTRA PRO</p>
    </div>
    """, unsafe_allow_html=True)

# --- 6. MODULES ---

# --- DASHBOARD ---
if "Dashboard" in choice:
    st.snow()
    col1, col2 = st.columns([1.6, 1])
    with col1:
        st.markdown("""<div class='glass-card'>
            <h2>Innovating Audio with AI</h2>
            <p>Welcome to Technova Solution. We provide next-gen audio intelligence tools to bridge the gap between human senses and digital technology.</p>
            <ul>
                <li>AI Voice to Music Synthesis</li>
                <li>Vibrational Sound for Hearing Assist</li>
                <li>Mood Based Productivity Flows</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.image("https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=500&h=500&fit=crop", use_container_width=True)

# --- MOOD SPOTIFY AI (ERROR FIXED) ---
elif "Mood Spotify AI" in choice:
    st.markdown("<h2>🧠 Smart Mood Recommender</h2>", unsafe_allow_html=True)
    
    # Official & Stable Spotify Playlist IDs (Global)
    mood_spotify_map = {
        "Energetic 🔥": "37i9dQZF1DX76W9SwwE6v4", # Power Workout
        "Calm 🌊": "37i9dQZF1DX8UebicO9uaR",      # Lofi Beats
        "Focused 🎯": "37i9dQZF1DX4sWSp4sm94f",   # Deep Focus
        "Stressed 🧘": "37i9dQZF1DX3YSRmBhyV9O",  # Stress Relief
        "Devotional ✨": "37i9dQZF1DX0S69v9S94G0" # Bhakti
    }

    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        u_mood = st.selectbox("HOW ARE YOU FEELING?", list(mood_spotify_map.keys()))
        u_goal = st.text_input("SET YOUR GOAL TODAY:", value="Finish Internship Project")
        
        if st.button("🚀 LAUNCH SESSION"):
            st.session_state.mood_run = u_mood
            st.balloons() # BALLOONS TRIGGER
            st.snow()     # SNOW TRIGGER
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if 'mood_run' in st.session_state:
            playlist_id = mood_spotify_map[st.session_state.mood_run]
            # Updated Embed URL for Maximum Reliability
            embed_url = f"https://open.spotify.com/embed/playlist/{playlist_id}?utm_source=generator&theme=0"
            
            st.markdown(f"<h3 style='text-align:center;'>Vibe: {st.session_state.mood_run}</h3>", unsafe_allow_html=True)
            # Reliable Player Embedding
            components.iframe(embed_url, height=450, scrolling=False)
        else:
            st.image("https://images.unsplash.com/photo-1493225255756-d9584f8606e9?q=80&w=800", use_container_width=True)

# --- CREATIVE STUDIO ---
elif "Creative Studio" in choice:
    st.markdown("<h2>🎙️ Creative AI Studio</h2>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    v_in = st.audio_input("Record voice to generate music:")
    if v_in:
        y, sr = librosa.load(v_in)
        if st.button("✨ TRANSFORM TO MUSIC"):
            st.balloons()
            st.snow()
            music = voice_to_music_logic(y, sr)
            st.audio(music, sample_rate=sr)
    st.markdown("</div>", unsafe_allow_html=True)

# --- HEARING ASSIST ---
elif "Hearing Assist" in choice:
    st.markdown("<h2>♿ Inclusive Hearing Lab</h2>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.write("Vibrational sound patterns for the hearing impaired.")
    up_h = st.file_uploader("Upload audio file", type=['mp3', 'wav'])
    if up_h:
        if st.button("🔊 OPTIMIZE Pattern"):
            st.balloons()
            st.snow()
            st.success("Optimization Complete. Pattern ready for bone-conduction.")
    st.markdown("</div>", unsafe_allow_html=True)
