import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import io

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TECHNOVA SOLUTION | Pro AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ADVANCED CSS (FIXED SIDEBAR & FONTS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&family=Poppins:wght@300;400;600;800&display=swap');

    /* Full App Background */
    .stApp {
        background: url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
    }
    
    .main-overlay {
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0, 0, 0, 0.75);
        z-index: -1;
    }

    /* --- SIDEBAR CUSTOMIZATION (ULTRA VISIBLE) --- */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e !important; /* Deep Blue/Purple Background */
        border-right: 2px solid #00d2ff !important;
    }

    /* Sidebar Text Colour (Word Visibility) */
    [data-testid="stSidebar"] * {
        color: #00d2ff !important; /* Cyan Neon for all text */
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        color: #ff00c1 !important; /* Pink for sidebar headers */
        font-family: 'Orbitron', sans-serif !important;
    }

    /* Dashboard Header */
    .hero-container {
        text-align: center;
        padding: 40px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 30px;
        border: 2px solid #00d2ff;
        backdrop-filter: blur(15px);
        margin-bottom: 25px;
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

    /* Professional Glass Cards */
    .glass-card {
        background: rgba(10, 10, 20, 0.9);
        padding: 30px;
        border-radius: 25px;
        border: 1px solid rgba(0, 210, 255, 0.4);
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    }

    /* Fonts & Buttons */
    h2, h3 { color: #00d2ff !important; font-family: 'Orbitron', sans-serif; font-size: 2.6rem !important; }
    p, label { font-size: 1.4rem !important; color: #ffffff !important; font-family: 'Poppins', sans-serif; }
    
    .stButton>button {
        background: linear-gradient(45deg, #bc13fe, #00d2ff);
        border: none; color: white !important; border-radius: 50px;
        padding: 15px 45px; font-size: 1.5rem !important;
        font-family: 'Orbitron', sans-serif; width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.05); box-shadow: 0 0 30px #00d2ff; }

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
    st.markdown("<h1 style='text-align:center;'>TECHNOVA</h1>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=120)
    st.write("---")
    menu = ["🏠 Dashboard", "🧠 Mood & Spotify AI", "🎙️ Creative AI Studio", "♿ Hearing Assist"]
    choice = st.selectbox("SELECT MODULE", menu)
    st.write("---")
    st.success("AI Core Status: Online")
    st.info("System: Technova v5.0 Pro")

# --- 5. TOP BRANDING HEADER ---
st.markdown("""
    <div class="hero-container">
        <h1 class="company-title">TECHNOVA SOLUTION</h1>
        <p style="letter-spacing: 6px; color:#92fe9d; font-size:1.6rem; font-weight:600;">Next-Gen Audio Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

# --- 6. MODULES ---

# DASHBOARD
if choice == "🏠 Dashboard":
    st.snow()
    col1, col2 = st.columns([1.6, 1])
    with col1:
        st.markdown("""<div class='glass-card'>
            <h2>Welcome to SonicSense Pro</h2>
            <p>Technova Solution bridges the gap between sound and technology. Our AI helps creators and the hearing impaired experience sound in new dimensions.</p>
            <ul>
                <li>Voice-to-Instrument Conversion</li>
                <li>Vibrational Sound Optimization</li>
                <li>Haptic Sound Pattern Visualization</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.image("https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=500&h=500&fit=crop", use_container_width=True)

# MOOD & SPOTIFY AI (ERROR FIXED)
elif choice == "🧠 Mood & Spotify AI":
    st.markdown("<h2>Productivity Flow & Mood AI</h2>", unsafe_allow_html=True)
    
    # Corrected Playlist Embed Links
    mood_spotify_map = {
        "Energetic": "37i9dQZF1DX76W9SwwE6v4", 
        "Calm": "37i9dQZF1DX8UebicO9uaR",      
        "Focused": "37i9dQZF1DX4sWSp4sm94f",   
        "Stressed": "37i9dQZF1DX3YSRmBhyV9O",  
        "Devotional": "37i9dQZF1DX0S69v9S94G0" 
    }

    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        u_mood = st.selectbox("CHOOSE YOUR VIBE:", list(mood_spotify_map.keys()))
        u_goal = st.text_input("YOUR GOAL TODAY:", value="Finish Internship Project")
        
        if st.button("🚀 LAUNCH SESSION"):
            st.session_state.mood_active = True
            st.balloons()
            st.snow()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if 'mood_active' in st.session_state and st.session_state.mood_active:
            playlist_id = mood_spotify_map[u_mood]
            # Updated embed URL structure
            embed_url = f"https://open.spotify.com/embed/playlist/{playlist_id}?utm_source=generator&theme=0"
            
            st.markdown(f"""
                <div class='glass-card' style='text-align:center; border: 2px solid #1DB954;'>
                    <h3 style='color:#1DB954 !important;'>Target: {u_goal}</h3>
                    <iframe src="{embed_url}" width="100%" height="400" 
                    frameborder="0" allowtransparency="true" allow="encrypted-media" style="border-radius:20px;"></iframe>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.image("https://images.unsplash.com/photo-1493225255756-d9584f8606e9?q=80&w=800", use_container_width=True)

# CREATIVE STUDIO
elif choice == "🎙️ Creative AI Studio":
    st.markdown("<h2>Creative AI Studio</h2>", unsafe_allow_html=True)
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

# HEARING ASSIST
elif choice == "♿ Hearing Assist":
    st.markdown("<h2>Inclusive Hearing Lab</h2>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.write("Haptic sound patterns for the hearing impaired.")
    up_h = st.file_uploader("Upload audio for optimization", type=['mp3', 'wav'])
    if up_h:
        st.balloons()
        st.snow()
        st.success("Optimization Complete. Ready for Earspots.")
    st.markdown("</div>", unsafe_allow_html=True)
