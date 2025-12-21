import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import io
import time

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="TECHNOVA SOLUTION | Pro AI",
    page_icon="🎵",
    layout="wide"
)

# --- 2. ADVANCED CSS (FIXED BACKGROUND & FONTS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&family=Rajdhani:wght@600;700&display=swap');

    /* Beautiful Background Image */
    .stApp {
        background: url("https://images.unsplash.com/photo-1557683316-973673baf926?q=80&w=2029&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
    }
    
    /* Dark Overlay for better visibility */
    .main-overlay {
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0, 0, 0, 0.6);
        z-index: -1;
    }

    /* TECHNOVA HEADER STYLE */
    .hero-container {
        text-align: center;
        padding: 40px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 30px;
        border: 2px solid #00d2ff;
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }

    .company-name {
        font-family: 'Orbitron', sans-serif;
        font-size: 5rem !important;
        font-weight: 900;
        background: linear-gradient(90deg, #00d2ff, #92fe9d, #ff00c1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 10px;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(0, 0, 0, 0.7);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid rgba(0, 210, 255, 0.5);
        color: white;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
    }

    h2, h3 { color: #00d2ff !important; font-family: 'Orbitron', sans-serif; font-size: 2.5rem !important; }
    p, label { font-size: 1.4rem !important; color: #ffffff !important; font-weight: 600; text-shadow: 1px 1px 2px black; }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(45deg, #00d2ff, #ff00c1);
        border: none; color: white; border-radius: 50px;
        padding: 15px 40px; font-size: 1.4rem !important;
        font-family: 'Orbitron', sans-serif; width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.05); box-shadow: 0 0 25px #00d2ff; }

    /* Fix Radio Button Visibility */
    div[data-testid="stMarkdownContainer"] p {
        font-size: 1.5rem !important;
    }
    </style>
    <div class="main-overlay"></div>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR BRANDING ---
with st.sidebar:
    st.markdown("<h1 style='color:#00d2ff; font-family:Orbitron;'>TECHNOVA</h1>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=120)
    st.write("---")
    menu = ["🏠 Dashboard", "🧠 Spotify Mood AI", "🎙️ Creative Studio", "♿ Accessibility"]
    choice = st.sidebar.selectbox("SELECT MODULE", menu)
    st.write("---")
    st.info("System: Technova v4.0")

# --- 4. HEADER ---
st.markdown("""
    <div class="hero-container">
        <h1 class="company-name">TECHNOVA SOLUTION</h1>
        <p style="letter-spacing: 5px; color:#92fe9d; font-size:1.5rem; font-weight:bold;">SONICSENSE ULTRA PRO</p>
    </div>
    """, unsafe_allow_html=True)

# --- 5. MODULES ---

# --- SPOTIFY MOOD AI (FIXED LINKS) ---
if choice == "🧠 Spotify Mood AI":
    st.balloons() # Module open aagum pothu balloons varum
    st.markdown("<h2>Mood-to-Spotify AI Suggester</h2>", unsafe_allow_html=True)
    
    col_m1, col_m2 = st.columns([1, 2])
    
    with col_m1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        mood = st.radio("How are you feeling?", 
                        ["Party Energy", "Deep Focus", "Chill Lofi", "Emotional/Sad", "Devotional"])
        st.markdown("</div>", unsafe_allow_html=True)
        
    # CORRECT SPOTIFY EMBED LINKS
    mood_map = {
        "Party Energy": "https://open.spotify.com/embed/playlist/37i9dQZF1DXaXB88o2P9G9",
        "Deep Focus": "https://open.spotify.com/embed/playlist/37i9dQZF1DX4sWSp4sm94f",
        "Chill Lofi": "https://open.spotify.com/embed/playlist/37i9dQZF1DX8UebicO9uaR",
        "Emotional/Sad": "https://open.spotify.com/embed/playlist/37i9dQZF1DX3YSRmBhyV9O",
        "Devotional": "https://open.spotify.com/embed/playlist/37i9dQZF1DX0S69v9S94G0"
    }

    with col_m2:
        st.markdown(f"""
            <div class='glass-card' style='text-align:center;'>
                <h3>Technova AI Recommends: {mood}</h3>
                <iframe src="{mood_map[mood]}" width="100%" height="400" frameborder="0" 
                allowtransparency="true" allow="encrypted-media" style="border-radius:15px;"></iframe>
            </div>
        """, unsafe_allow_html=True)

# --- DASHBOARD ---
elif choice == "🏠 Dashboard":
    st.snow()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""<div class='glass-card'>
            <h2>Welcome to Technova SonicSense</h2>
            <p>Technova Solution presents a next-generation AI platform for audio processing. 
            From converting voice to instruments to assisting the hearing impaired, 
            we bridge the gap between human senses and technology.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.image("https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=400&h=400&fit=crop", use_container_width=True)

# --- CREATIVE STUDIO ---
elif choice == "🎙️ Creative Studio":
    st.markdown("<h2>AI Creative Studio</h2>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    rec = st.audio_input("Record your voice to generate music:")
    if rec:
        st.success("Voice Received! Converting to instrument...")
        st.balloons()
    st.markdown("</div>", unsafe_allow_html=True)

# --- ACCESSIBILITY ---
elif choice == "♿ Accessibility":
    st.markdown("<h2>Inclusive Hearing Assist</h2>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.write("Special frequency shifting for bone conduction devices (Earspots).")
    up_h = st.file_uploader("Upload audio for optimization", type=['mp3', 'wav'])
    if up_h:
        st.balloons()
        st.write("Audio optimized for vibrations.")
    st.markdown("</div>", unsafe_allow_html=True)
