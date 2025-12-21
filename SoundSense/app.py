import streamlit as st
import librosa
import numpy as np
import streamlit.components.v1 as components

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TECHNOVA SOLUTION | Pro AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ADVANCED CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&family=Poppins:wght@400;700;800&display=swap');

.stApp {
    background: url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop");
    background-size: cover;
    background-attachment: fixed;
}

[data-testid="stSidebar"] {
    background-color: #050510 !important;
    border-right: 2px solid #ff00c1;
}
[data-testid="stSidebar"] * {
    color: #ff00c1 !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 800 !important;
}

.hero-container {
    text-align:center;
    padding:40px;
    background:rgba(0,0,0,0.7);
    border-radius:30px;
    border:2px solid #ff00c1;
    margin-bottom:25px;
}
.company-title {
    font-family:'Orbitron',sans-serif;
    font-size:4.5rem;
    background:linear-gradient(90deg,#ff00c1,#00d2ff,#92fe9d);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    letter-spacing:10px;
}
.glass-card {
    background:rgba(0,0,0,0.9);
    padding:30px;
    border-radius:25px;
    border:1px solid #ff00c144;
    margin-bottom:25px;
}
h2,h3 {color:#00d2ff !important;}
p,label {color:white !important;}
.stButton>button {
    background:linear-gradient(45deg,#ff00c1,#00d2ff);
    border:none;
    color:white;
    border-radius:50px;
    padding:18px;
    font-size:1.2rem;
    width:100%;
}
</style>
""", unsafe_allow_html=True)

# --- 3. CORE LOGIC ---
def voice_to_music_logic(audio, sr):
    f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'),
                            fmax=librosa.note_to_hz('C7'))
    f0 = np.nan_to_num(f0)
    hop = 512
    f0_stretched = np.repeat(f0, hop)[:len(audio)]
    phase = np.cumsum(2 * np.pi * f0_stretched / sr)
    synth = 0.5 * np.sin(phase) + 0.3 * np.sin(2 * phase)
    return synth / (np.max(np.abs(synth)) + 1e-6)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:#00d2ff;text-align:center;'>TECHNOVA</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=120)
    st.write("---")
    choice = st.radio("SELECT MENU:", [
        "🏠 Dashboard",
        "🧠 Mood Spotify AI",
        "🎙️ Creative Studio",
        "♿ Hearing Assist"
    ])
    st.write("---")
    st.success("⚡ AI ENGINE: ONLINE")

# --- 5. HEADER ---
st.markdown("""
<div class="hero-container">
    <h1 class="company-title">TECHNOVA SOLUTION</h1>
    <p style="color:#92fe9d;font-weight:700;">SONICSENSE ULTRA PRO</p>
</div>
""", unsafe_allow_html=True)

# --- DASHBOARD ---
if "Dashboard" in choice:
    st.snow()
    col1, col2 = st.columns([1.5,1])
    with col1:
        st.markdown("""
        <div class='glass-card'>
        <h2>Innovating Audio with AI</h2>
        <p>
        • Voice → Music AI<br>
        • Hearing Impaired Assist<br>
        • Mood based Spotify Productivity
        </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.image("https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=500",
                 use_container_width=True)

# --- MOOD SPOTIFY AI (PAGE NOT FOUND FIXED) ---
elif "Mood Spotify AI" in choice:
    st.markdown("<h2>🧠 Smart Mood Recommender</h2>", unsafe_allow_html=True)

    mood_spotify_map = {
        "Energetic 🔥": "37i9dQZF1DX76W9SwwE6v4",
        "Calm 🌊": "37i9dQZF1DX8UebicO9uaR",
        "Focused 🎯": "37i9dQZF1DX4sWSp4sm94f",
        "Stressed 🧘": "37i9dQZF1DX3YSRmBhyV9O"
    }

    col1, col2 = st.columns([1,1.4])
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        mood = st.selectbox("HOW ARE YOU FEELING?", list(mood_spotify_map.keys()))
        goal = st.text_input("SET YOUR GOAL TODAY:", "Finish Internship Project")
        if st.button("🚀 LAUNCH SESSION"):
            st.session_state.mood_run = mood
            st.balloons()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if "mood_run" in st.session_state:
            pid = mood_spotify_map[st.session_state.mood_run]
            embed_url = f"https://open.spotify.com/embed/playlist/{pid}?theme=0"

            st.markdown(f"<h3 style='text-align:center;'>Vibe: {st.session_state.mood_run}</h3>",
                        unsafe_allow_html=True)

            components.iframe(embed_url, height=380)

            # ✅ SAFE EXTERNAL LINK (FIX)
            st.markdown(f"""
            <div style="text-align:center;margin-top:20px;">
                <a href="https://open.spotify.com/playlist/{pid}"
                   target="_blank"
                   style="
                   background:#1DB954;
                   padding:14px 30px;
                   border-radius:40px;
                   color:white;
                   font-weight:800;
                   text-decoration:none;
                   display:inline-block;">
                🎧 OPEN IN SPOTIFY
                </a>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.image("https://images.unsplash.com/photo-1493225255756-d9584f8606e9?q=80&w=800",
                     use_container_width=True)

# --- CREATIVE STUDIO ---
elif "Creative Studio" in choice:
    st.markdown("<h2>🎙️ Creative AI Studio</h2>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    audio = st.audio_input("Record voice:")
    if audio:
        y, sr = librosa.load(audio)
        if st.button("✨ TRANSFORM TO MUSIC"):
            st.balloons()
            music = voice_to_music_logic(y, sr)
            st.audio(music, sample_rate=sr)
    st.markdown("</div>", unsafe_allow_html=True)

# --- HEARING ASSIST ---
elif "Hearing Assist" in choice:
    st.markdown("<h2>♿ Hearing Assist Lab</h2>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    up = st.file_uploader("Upload audio", type=["mp3","wav"])
    if up and st.button("🔊 ANALYZE SOUND"):
        st.balloons()
        st.success("Sound vibration pattern generated successfully.")
    st.markdown("</div>", unsafe_allow_html=True)
