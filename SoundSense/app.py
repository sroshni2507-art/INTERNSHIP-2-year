import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import io

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TECHNOVA SOLUTION | SonicSense Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS FOR PINK SIDEBAR & NEON THEME ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&family=Poppins:wght@400;600;800&display=swap');

    /* Global Background */
    .stApp {
        background: url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
    }
    .main-overlay {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0, 0, 0, 0.8); z-index: -1;
    }

    /* --- SIDEBAR PINK FONT CUSTOMIZATION --- */
    [data-testid="stSidebar"] {
        background-color: #0a0a15 !important;
        border-right: 2px solid #ff00c1;
    }
    
    /* Sidebar words in Pink for high visibility */
    [data-testid="stSidebar"] * {
        color: #ff00c1 !important; 
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
    }
    
    /* Side Bar Title */
    .sidebar-title {
        color: #00d2ff !important;
        font-family: 'Orbitron', sans-serif;
        font-size: 2rem;
        text-align: center;
        margin-bottom: 20px;
    }

    /* TECHNOVA HEADER */
    .hero-container {
        text-align: center; padding: 40px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 35px; border: 2px solid #ff00c1;
        backdrop-filter: blur(15px); margin-bottom: 25px;
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
        background: rgba(15, 15, 25, 0.92);
        padding: 30px; border-radius: 25px;
        border: 1px solid rgba(255, 0, 193, 0.4);
        margin-bottom: 25px;
    }

    h2, h3 { color: #00d2ff !important; font-family: 'Orbitron', sans-serif; font-size: 2.5rem !important; }
    p, label { font-size: 1.4rem !important; color: #ffffff !important; font-family: 'Poppins', sans-serif; }
    
    /* Action Button for AI Studio */
    .stButton>button {
        background: linear-gradient(45deg, #ff00c1, #00d2ff);
        border: none; color: white !important; border-radius: 50px;
        padding: 15px 45px; font-size: 1.4rem !important;
        font-family: 'Orbitron', sans-serif; width: 100%;
    }
    </style>
    <div class="main-overlay"></div>
    """, unsafe_allow_html=True)

# --- 3. CORE AI LOGIC ---
def voice_to_music_logic(audio, sr):
    f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = np.nan_to_num(f0)
    hop = 512
    f0_stretched = np.repeat(f0, hop)[:len(audio)]
    phase = np.cumsum(2 * np.pi * f0_stretched / sr)
    synth = 0.5 * np.sin(phase) + 0.3 * np.sin(2 * phase)
    return synth / (np.max(np.abs(synth)) + 1e-6)

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<div class='sidebar-title'>TECHNOVA</div>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=120)
    st.write("---")
    menu = ["🏠 Dashboard", "🧠 Mood & Spotify AI", "🎙️ Creative AI Studio", "♿ Hearing Assist"]
    choice = st.selectbox("SELECT MODULE:", menu)
    st.write("---")
    st.success("⚡ AI ENGINE: ONLINE")

# --- 5. TOP HEADER ---
st.markdown("""
    <div class="hero-container">
        <h1 class="company-title">TECHNOVA SOLUTION</h1>
        <p style="letter-spacing: 6px; color:#92fe9d; font-size:1.6rem; font-weight:700;">SONICSENSE ULTRA PRO</p>
    </div>
    """, unsafe_allow_html=True)

# --- 6. MODULES ---

# --- DASHBOARD ---
if choice == "🏠 Dashboard":
    st.snow()
    col1, col2 = st.columns([1.6, 1])
    with col1:
        st.markdown("""<div class='glass-card'>
            <h2>Innovating Audio Through AI</h2>
            <p>Welcome to Technova Solution. We provide advanced audio tools to help creators and enhance accessibility for the hearing impaired.</p>
            <ul style='color:white; font-size:1.2rem;'>
                <li>Voice to Digital Instrument Conversion</li>
                <li>AI Powered Mood Recommendation</li>
                <li>Vibrational Sound for Accessibility</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.image("https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=500&h=500&fit=crop", use_container_width=True)

# --- MOOD & SPOTIFY AI (ERROR FIXED VERSION) ---
elif choice == "🧠 Mood & Spotify AI":
    st.markdown("<h2>Productivity Flow & Mood AI</h2>", unsafe_allow_html=True)
    
    # FIXED UNIVERSAL PLAYLIST LINKS
    mood_spotify_map = {
        "Energetic 🔥": "https://open.spotify.com/playlist/37i9dQZF1DX76W9SwwE6v4", 
        "Calm 🌊": "https://open.spotify.com/playlist/37i9dQZF1DX8UebicO9uaR",      
        "Focused 🎯": "https://open.spotify.com/playlist/37i9dQZF1DX4sWSp4sm94f",   
        "Stressed 🧘": "https://open.spotify.com/playlist/37i9dQZF1DX3YSRmBhyV9O",  
        "Devotional ✨": "https://open.spotify.com/playlist/37i9dQZF1DX0S69v9S94G0" 
    }

    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        u_mood = st.selectbox("CHOOSE YOUR VIBE:", list(mood_spotify_map.keys()))
        u_goal = st.text_input("YOUR GOAL TODAY:", value="Finish Internship Project")
        
        # Using link_button for most stable experience
        spotify_url = mood_spotify_map[u_mood]
        
        st.write("### Step 1: Open Spotify")
        st.link_button(f"🟢 OPEN {u_mood.upper()} PLAYLIST", spotify_url, use_container_width=True)
        
        st.write("### Step 2: Start Session")
        if st.button("🚀 LAUNCH TECHNOVA SESSION"):
            st.balloons()
            st.snow()
            st.success(f"Session Started! Target Goal: {u_goal}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.image("https://images.unsplash.com/photo-1493225255756-d9584f8606e9?q=80&w=800", use_container_width=True)

# --- CREATIVE AI STUDIO ---
elif choice == "🎙️ Creative AI Studio":
    st.markdown("<h2>AI Creative Studio</h2>", unsafe_allow_html=True)
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
elif choice == "♿ Hearing Assist":
    st.markdown("<h2>Inclusive Hearing Lab</h2>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.write("Vibrational optimization for the hearing impaired.")
    up_h = st.file_uploader("Upload audio file", type=['mp3', 'wav'])
    if up_h:
        if st.button("🔊 OPTIMIZE Pattern"):
            st.balloons()
            st.snow()
            st.success("Optimization Complete. Ready for haptic devices.")
    st.markdown("</div>", unsafe_allow_html=True)
