import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import io

# ================= 1. PAGE CONFIG =================
st.set_page_config(
    page_title="TECHNOVA SOLUTION | SonicSense Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= 2. ADVANCED NEON CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;900&family=Poppins:wght@300;400;700&display=swap');

/* Main Background */
.stApp {
    background: url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop");
    background-size: cover;
    background-attachment: fixed;
}

/* Dark Blur Overlay */
.main-overlay {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0, 0, 0, 0.78); z-index: -1;
}

/* --- SIDEBAR PINK NEON --- */
[data-testid="stSidebar"] {
    background: #050510 !important;
    border-right: 2px solid #ff00c1 !important;
}
[data-testid="stSidebar"] * {
    color: #ff00c1 !important; /* PINK WORDS */
    font-family: 'Poppins', sans-serif;
    font-weight: 800 !important;
    font-size: 1.1rem;
}

/* Glassmorphic Cards */
.glass {
    background: rgba(10, 10, 20, 0.9);
    padding: 35px;
    border-radius: 30px;
    border: 1px solid rgba(0, 210, 255, 0.3);
    box-shadow: 0 15px 45px rgba(0,0,0,0.8);
    margin-bottom: 25px;
}

/* TECHNOVA LOGO HEADER */
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 4.5rem !important;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #ff00c1, #00d2ff, #92fe9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 12px;
    margin-bottom: 0;
    animation: glow 3s infinite alternate;
}

@keyframes glow {
    from { text-shadow: 0 0 10px rgba(255, 0, 193, 0.5); }
    to { text-shadow: 0 0 30px rgba(0, 210, 255, 0.8); }
}

/* Visible Font Style */
h2, h3 { color: #00d2ff !important; font-family: 'Orbitron', sans-serif; }
p, label { font-size: 1.3rem !important; color: white !important; font-family: 'Poppins', sans-serif; }

/* Premium Buttons */
.stButton>button {
    background: linear-gradient(45deg, #ff00c1, #00d2ff);
    color: white !important;
    border-radius: 50px;
    padding: 15px 45px;
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    border: none;
    box-shadow: 0 10px 30px rgba(255, 0, 193, 0.4);
    transition: 0.3s;
    width: 100%;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 40px #ff00c1;
}

</style>
<div class="main-overlay"></div>
""", unsafe_allow_html=True)

# ================= 3. AUDIO LOGIC =================
def voice_to_music(audio_data, sr):
    f0, _, _ = librosa.pyin(audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = np.nan_to_num(f0)
    phase = np.cumsum(2 * np.pi * f0 / sr)
    music = 0.5 * np.sin(phase) + 0.3 * np.sin(2 * phase)
    return music / (np.max(np.abs(music)) + 1e-6)

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>TECHNOVA</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=100)
    st.write("---")
    menu = st.radio(
        "NAVIGATE",
        ["🏠 Dashboard", "🧠 Mood Spotify AI", "🎙️ Creative Studio", "♿ Hearing Assist"]
    )
    st.write("---")
    st.success("⚡ AI ENGINE : ONLINE")
    st.info("Ver: 5.0 Ultra Pro")

# ================= 5. HEADER =================
st.markdown("<h1 class='hero-title'>TECHNOVA SOLUTION</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; letter-spacing:5px; color:#92fe9d; font-weight:bold;'>NEXT-GEN AUDIO INTELLIGENCE</p>", unsafe_allow_html=True)

# ================= 6. MODULES =================

# --- DASHBOARD ---
if menu == "🏠 Dashboard":
    st.snow()
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("""
        <div class='glass'>
            <h2>Innovating sound for everyone.</h2>
            <p>Welcome to <b>Technova Solution</b>. Our SonicSense platform uses AI to bridge the gap between human senses and digital audio.</p>
            <div style='margin-top:20px;'>
                <p>✅ <b>AI Music</b>: Convert voice to instruments.</p>
                <p>✅ <b>Mood AI</b>: Productivity focused sessions.</p>
                <p>✅ <b>Inclusive</b>: Accessibility for hearing impaired.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.image("https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=600&h=600&fit=crop", use_container_width=True)

# --- MOOD SPOTIFY AI (STABLE) ---
elif menu == "🧠 Mood Spotify AI":
    st.markdown("<div class='glass'><h3>🧠 Mood-Based Smart Suggestion</h3></div>", unsafe_allow_html=True)
    
    # Correct Official Playlist IDs
    mood_map = {
        "Energetic 🔥": "37i9dQZF1DX76W9SwwE6v4",
        "Calm 🌊": "37i9dQZF1DX8UebicO9uaR",
        "Focused 🎯": "37i9dQZF1DX4sWSp4sm94f",
        "Stressed 🧘": "37i9dQZF1DX3YSRmBhyV9O",
        "Devotional ✨": "37i9dQZF1DX0S69v9S94G0"
    }

    c1, c2 = st.columns([1, 1.3])
    with c1:
        mood = st.selectbox("HOW ARE YOU FEELING?", list(mood_map.keys()))
        goal = st.text_input("YOUR GOAL TODAY", "Finish Internship Project")
        if st.button("🚀 LAUNCH SESSION"):
            st.session_state.active_mood = mood
            st.balloons()
            st.snow()

    with c2:
        if 'active_mood' in st.session_state:
            playlist_id = mood_map[st.session_state.active_mood]
            # Official Spotify Embed format
            embed_url = f"https://open.spotify.com/embed/playlist/{playlist_id}?utm_source=generator&theme=0"
            
            st.markdown(f"<h4>Target: {goal}</h4>", unsafe_allow_html=True)
            components.iframe(embed_url, height=450, scrolling=False)
        else:
            st.image("https://images.unsplash.com/photo-1493225255756-d9584f8606e9?q=80&w=800", use_container_width=True)

# --- CREATIVE STUDIO ---
elif menu == "🎙️ Creative Studio":
    st.markdown("<div class='glass'><h3>🎙️ Creative AI Studio</h3><p>Sing or speak to convert it into a digital melody.</p></div>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        audio = st.audio_input("Record your voice")
    
    with col_b:
        if audio:
            y, sr = librosa.load(audio)
            if st.button("✨ TRANSFORM TO MUSIC"):
                st.balloons()
                music = voice_to_music(y, sr)
                st.audio(music, sample_rate=sr)
                st.success("Technova AI composition complete!")

# --- HEARING ASSIST ---
elif menu == "♿ Hearing Assist":
    st.markdown("<div class='glass'><h3>♿ Inclusive Hearing Lab</h3><p>Frequency shifting & visual vibration patterns for bone-conduction devices.</p></div>", unsafe_allow_html=True)
    
    up_file = st.file_uploader("Upload audio for pattern optimization", type=["wav","mp3"])
    if up_file:
        y, sr = librosa.load(up_file)
        if st.button("🔊 OPTIMIZE Pattern"):
            st.balloons()
            st.snow()
            st.success("Optimization Complete. Ready for Earspots haptic feedback.")
            
            # Simple Pulse Graph
            rms = librosa.feature.rms(y=y)[0]
            fig, ax = plt.subplots(figsize=(10, 2), facecolor='black')
            ax.plot(rms, color='#00d2ff', linewidth=2)
            ax.fill_between(range(len(rms)), rms, color='#ff00c1', alpha=0.3)
            ax.set_axis_off()
            st.pyplot(fig)
