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

# --- 2. PREMIUM CSS (VISIBLE SIDEBAR & CLEAN FONTS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&family=Poppins:wght@400;600;800&display=swap');

    /* Global App Style */
    .stApp {
        background: url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
    }
    .main-overlay {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0, 0, 0, 0.78); z-index: -1;
    }

    /* --- COLORFUL SIDEBAR (FIXED VISIBILITY) --- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
        border-right: 3px solid #00d2ff;
    }
    /* Sidebar Text & Icons Colour */
    [data-testid="stSidebar"] * {
        color: #92fe9d !important; /* Bright Mint Green for words */
        font-family: 'Poppins', sans-serif !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebarNav"] span {
        color: #ffffff !important;
        background: rgba(0, 210, 255, 0.2);
        padding: 5px 15px;
        border-radius: 10px;
    }

    /* TECHNOVA HEADER */
    .hero-container {
        text-align: center; padding: 40px;
        background: rgba(255, 255, 255, 0.04);
        border-radius: 35px; border: 2px solid #00d2ff;
        backdrop-filter: blur(15px); margin-bottom: 25px;
        box-shadow: 0 0 40px rgba(0, 210, 255, 0.3);
    }
    .company-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 5rem !important;
        font-weight: 900;
        background: linear-gradient(90deg, #ff00c1, #00d2ff, #92fe9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 12px;
        animation: glow 3s infinite alternate;
    }
    @keyframes glow {
        from { text-shadow: 0 0 10px #ff00c1; }
        to { text-shadow: 0 0 35px #00d2ff; }
    }

    /* Content Cards */
    .glass-card {
        background: rgba(10, 10, 20, 0.92);
        padding: 30px; border-radius: 25px;
        border: 1px solid rgba(0, 210, 255, 0.5);
        margin-bottom: 25px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.8);
    }

    h2, h3 { color: #00d2ff !important; font-family: 'Orbitron', sans-serif; font-size: 2.7rem !important; }
    p, label { font-size: 1.5rem !important; color: #ffffff !important; font-family: 'Poppins', sans-serif; font-weight: 500; }
    
    /* Neon Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #bc13fe, #00d2ff);
        border: none; color: white !important; border-radius: 50px;
        padding: 18px 45px; font-size: 1.5rem !important;
        font-family: 'Orbitron', sans-serif; width: 100%;
        transition: 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .stButton>button:hover { transform: scale(1.05); box-shadow: 0 0 40px #00d2ff; }
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

# --- 4. COLORFUL SIDEBAR MENU ---
with st.sidebar:
    st.markdown("<h2 style='color:#ff00c1 !important; text-align:center;'>TECHNOVA</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=120)
    st.write("---")
    
    # Adding Icons to Menu Items
    menu_options = {
        "🏠 Dashboard": "Home",
        "🧠 Mood & Spotify AI": "Spotify",
        "🎙️ Creative AI Studio": "Studio",
        "♿ Hearing Assist": "Assist",
        "🌈 Sensory Pulse": "Sensory"
    }
    
    choice = st.radio("SELECT MODULE:", list(menu_options.keys()))
    
    st.write("---")
    st.success("⚡ AI Engine: Online")
    st.info("💎 Status: Premium v5.0")

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
            <h2>Innovating Sound for Everyone</h2>
            <p>Welcome to Technova Solution. We bridge the gap between human senses and AI technology through advanced audio intelligence.</p>
            <ul style="color:white; font-size:1.3rem;">
                <li>Voice-to-Instrument Synthesis</li>
                <li>Vibrational Audio for Accessibility</li>
                <li>Mood-Driven Productivity Loops</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.image("https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=500&h=500&fit=crop", use_container_width=True)

# --- MOOD & SPOTIFY AI (FIXED ERROR) ---
elif choice == "🧠 Mood & Spotify AI":
    st.markdown("<h2>Productivity Flow & Mood AI</h2>", unsafe_allow_html=True)
    
    # Updated Verified Spotify IDs
    mood_spotify_map = {
        "Energetic 🔥": "37i9dQZF1DX76W9SwwE6v4", 
        "Calm 🌊": "37i9dQZF1DX8UebicO9uaR",      
        "Focused 🎯": "37i9dQZF1DX4sWSp4sm94f",   
        "Stressed 🧘": "37i9dQZF1DX3YSRmBhyV9O",  
        "Devotional ✨": "37i9dQZF1DX0S69v9S94G0" 
    }

    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        u_mood = st.selectbox("CHOOSE YOUR VIBE:", list(mood_spotify_map.keys()))
        u_goal = st.text_input("YOUR GOAL TODAY:", value="Finish Internship Project")
        
        if st.button("🚀 LAUNCH TECHNOVA SESSION"):
            st.session_state.mood_active = True
            st.balloons()
            st.snow()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if 'mood_active' in st.session_state and st.session_state.mood_active:
            # FIX: Using st.components.v1.iframe for stable embedding
            playlist_id = mood_spotify_map[u_mood]
            embed_url = f"https://open.spotify.com/embed/playlist/{playlist_id}?utm_source=generator"
            
            st.markdown(f"<h3 style='text-align:center;'>Target: {u_goal}</h3>", unsafe_allow_html=True)
            components.iframe(embed_url, height=450, scrolling=False)
        else:
            st.image("https://images.unsplash.com/photo-1493225255756-d9584f8606e9?q=80&w=800", use_container_width=True)

# --- CREATIVE AI STUDIO ---
elif choice == "🎙️ Creative AI Studio":
    st.markdown("<h2>AI Creative Studio</h2>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    v_in = st.audio_input("Record your voice/singing:")
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
    st.write("Special frequency shifting for bone-conduction vibrations.")
    up_h = st.file_uploader("Upload audio for haptic optimization", type=['mp3', 'wav'])
    if up_h:
        y, sr = librosa.load(up_h)
        if st.button("🔊 OPTIMIZE Pattern"):
            st.balloons()
            st.snow()
            y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=-8)
            st.audio(y_shift * 1.8, sample_rate=sr)
            st.success("Optimization Complete. Ready for Earspots patterns.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- SENSORY PULSE ---
elif choice == "🌈 Sensory Pulse":
    st.markdown("<h2>Sensory Pulse Visualization</h2>", unsafe_allow_html=True)
    up_v = st.file_uploader("Upload sound to see waveform", type=['mp3','wav'])
    if up_v:
        y, sr = librosa.load(up_v)
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 4), facecolor='black')
        ax.plot(y[::100], color='#00d2ff', alpha=0.8)
        ax.set_axis_off()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
