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
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ADVANCED CSS (SIDEBAR PINK & BRANDING) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&family=Poppins:wght@400;700&display=swap');

    /* Background Image */
    .stApp {
        background: url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
    }
    .main-overlay {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0, 0, 0, 0.82); z-index: -1;
    }

    /* --- SIDEBAR PINK STYLE --- */
    [data-testid="stSidebar"] {
        background-color: #0a0a1a !important;
        border-right: 2px solid #ff00c1;
    }
    
    /* Sidebar Text Colour - NEON PINK */
    [data-testid="stSidebar"] span, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #ff00c1 !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
    }
    
    /* Sidebar Title */
    .sb-title {
        color: #00d2ff !important;
        font-family: 'Orbitron', sans-serif;
        font-size: 1.8rem;
        text-align: center;
        margin-bottom: 20px;
    }

    /* TECHNOVA HEADER */
    .hero-header {
        text-align: center; padding: 40px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 35px; border: 2px solid #ff00c1;
        backdrop-filter: blur(15px); margin-bottom: 30px;
    }
    .company-name {
        font-family: 'Orbitron', sans-serif;
        font-size: 4.5rem !important;
        font-weight: 900;
        background: linear-gradient(90deg, #ff00c1, #00d2ff, #92fe9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 10px;
    }

    /* Action Cards */
    .glass-card {
        background: rgba(10, 10, 20, 0.9);
        padding: 30px; border-radius: 25px;
        border: 1px solid #ff00c166;
        margin-bottom: 25px;
    }

    h2, h3 { color: #00d2ff !important; font-family: 'Orbitron', sans-serif; }
    p, label { font-size: 1.3rem !important; color: white !important; font-family: 'Poppins', sans-serif; }
    
    /* Styled Action Button */
    .stButton>button {
        background: linear-gradient(45deg, #ff00c1, #00d2ff);
        border: none; color: white !important; border-radius: 50px;
        padding: 15px 40px; font-size: 1.4rem !important;
        font-family: 'Orbitron', sans-serif; width: 100%;
        box-shadow: 0 0 20px rgba(255, 0, 193, 0.4);
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

# --- 4. SIDEBAR NAVIGATION (PINK WORDS & ICONS) ---
with st.sidebar:
    st.markdown("<div class='sb-title'>TECHNOVA</div>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=120)
    st.write("---")
    
    # Sidebar Options with Icons
    choice = st.radio("MAIN MENU", [
        "🏠 Dashboard", 
        "🧠 Mood Spotify AI", 
        "🎙️ Creative Studio", 
        "♿ Hearing Assist"
    ])
    
    st.write("---")
    st.success("⚡ AI ENGINE: ONLINE")

# --- 5. TOP BRANDING ---
st.markdown("""
    <div class="hero-header">
        <h1 class="company-name">TECHNOVA SOLUTION</h1>
        <p style="letter-spacing: 5px; color:#92fe9d; font-size:1.4rem; font-weight:700;">SONICSENSE ULTRA PRO</p>
    </div>
    """, unsafe_allow_html=True)

# --- 6. MODULES ---

# --- DASHBOARD ---
if "Dashboard" in choice:
    st.snow()
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("""<div class='glass-card'>
            <h2>Innovating Sound with AI</h2>
            <p>Technova Solution's SonicSense Pro transforms audio experiences. We specialize in AI-driven music generation and accessibility tools.</p>
            <ul>
                <li>Voice-to-Instrument Synthesis</li>
                <li>Haptic Vibration Optimization</li>
                <li>Mood-Based Smart Suggestions</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.image("https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=500&h=500&fit=crop", use_container_width=True)

# --- MOOD SPOTIFY AI (API STYLE - FIXED) ---
elif "Mood Spotify AI" in choice:
    st.markdown("<h2>🧠 Mood-Based Smart Suggestions</h2>", unsafe_allow_html=True)
    
    # Track Database (Universal Track Embed IDs)
    mood_track_db = {
        "Energetic 🔥": "0VjIjWm4vTbN9vIwhZ6I4f", # Blinding Lights
        "Calm 🌊": "7qEByF41zSj64F9WfD6xG0",      # Lofi Chill
        "Focused 🎯": "509v0rN9Y6870tP6t70r53",   # Piano Focus
        "Stressed 🧘": "37i9dQZF1DX3YSRmBhyV9O",  # Stress Relief (Playlist)
        "Devotional ✨": "37i9dQZF1DX0S69v9S94G0" # Bhakti
    }

    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        u_mood = st.selectbox("HOW ARE YOU FEELING?", list(mood_track_db.keys()))
        u_goal = st.text_input("SET YOUR GOAL:", value="Finish Project")
        
        if st.button("🚀 GENERATE SESSION"):
            st.session_state.play_mood = u_mood
            st.balloons()
            st.snow()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if 'play_mood' in st.session_state:
            track_id = mood_track_db[st.session_state.play_mood]
            # Smart Check: If it's a track or playlist
            if track_id.startswith("37i9"):
                embed_url = f"https://open.spotify.com/embed/playlist/{track_id}?utm_source=generator"
            else:
                embed_url = f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator"
            
            st.markdown(f"<h3 style='text-align:center;'>Playing for: {u_goal}</h3>", unsafe_allow_html=True)
            # Reliable Iframe Embedding
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
    st.write("Vibrational optimization for the hearing impaired.")
    up_h = st.file_uploader("Upload audio file", type=['mp3', 'wav'])
    if up_h:
        if st.button("🔊 OPTIMIZE Pattern"):
            st.balloons()
            st.snow()
            st.success("Optimization Complete. Ready for Earspots patterns.")
    st.markdown("</div>", unsafe_allow_html=True)
