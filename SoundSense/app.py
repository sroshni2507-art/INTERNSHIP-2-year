import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import io
import time

# --- 1. PAGE CONFIG & BACKGROUND SETTINGS ---
st.set_page_config(
    page_title="TECHNOVA SOLUTION | SonicSense Pro",
    page_icon="🎵",
    layout="wide"
)

# --- 2. ADVANCED CSS (Full App Look) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&family=Rajdhani:wght@600;700&display=swap');

    /* Background Image and Overlay */
    .stApp {
        background: url("https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-attachment: fixed;
    }
    
    /* Overlay to make content readable */
    .main-overlay {
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0, 0, 0, 0.7);
        z-index: -1;
    }

    /* TECHNOVA HERO HEADER */
    .hero-container {
        text-align: center;
        padding: 50px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 30px;
        border: 2px solid #00d2ff;
        backdrop-filter: blur(15px);
        margin-bottom: 30px;
    }

    .company-name {
        font-family: 'Orbitron', sans-serif;
        font-size: 5.5rem !important;
        font-weight: 900;
        background: linear-gradient(90deg, #ff00c1, #00d2ff, #92fe9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 12px;
        animation: glow 3s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from { text-shadow: 0 0 10px #ff00c1; }
        to { text-shadow: 0 0 30px #00d2ff; }
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(0, 0, 0, 0.6);
        padding: 30px;
        border-radius: 20px;
        border: 1px solid rgba(0, 210, 255, 0.4);
        color: white;
        margin-bottom: 20px;
    }

    /* Huge Visible Font for Content */
    h2, h3 { color: #00d2ff !important; font-family: 'Orbitron', sans-serif; font-size: 2.2rem !important; }
    p, label { font-size: 1.4rem !important; color: #ffffff !important; font-weight: 600; }

    /* Custom Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #ff00c1, #00d2ff);
        border: none; color: white; border-radius: 50px;
        padding: 15px 40px; font-size: 1.3rem !important;
        font-family: 'Orbitron', sans-serif; width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.05); box-shadow: 0 0 30px #ff00c1; }

    </style>
    <div class="main-overlay"></div>
    """, unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---
def transform_voice(audio, sr):
    f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = np.nan_to_num(f0)
    hop = 512
    f0_stretched = np.repeat(f0, hop)[:len(audio)]
    phase = np.cumsum(2 * np.pi * f0_stretched / sr)
    synth = 0.5 * np.sin(phase) + 0.3 * np.sin(2 * phase)
    return synth / (np.max(np.abs(synth)) + 1e-6)

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h1 style='color:#00d2ff; font-family:Orbitron;'>TECHNOVA</h1>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=100)
    st.write("---")
    menu = ["🏠 Dashboard", "🧠 Spotify Mood AI", "🎙️ Creative Studio", "🌈 Sensory Pulse", "♿ Accessibility"]
    choice = st.sidebar.selectbox("SELECT MODULE", menu)
    st.write("---")
    st.info("System: Technova v4.0")

# --- 5. HEADER ---
st.markdown("""
    <div class="hero-container">
        <h1 class="company-name">TECHNOVA SOLUTION</h1>
        <p style="letter-spacing: 5px; color:#92fe9d; font-size:1.5rem;">SONICSENSE ULTRA PRO</p>
    </div>
    """, unsafe_allow_html=True)

# --- 6. MODULES ---

if choice == "🏠 Dashboard":
    st.balloons()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""<div class='glass-card'>
            <h2>Welcome to Technova SonicSense</h2>
            <p>Experience the intersection of AI and Sound. Our platform provides high-end audio solutions 
            for creators and accessibility tools for the hearing impaired.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.image("https://images.unsplash.com/photo-1470225620780-dba8ba36b745?q=80&w=400&h=400&fit=crop", use_container_width=True)

elif choice == "🧠 Spotify Mood AI":
    st.snow()
    st.markdown("<h2>Mood-to-Spotify AI Suggester</h2>", unsafe_allow_html=True)
    
    col_m1, col_m2 = st.columns([1, 2])
    
    with col_m1:
        st.write("How are you feeling?")
        mood = st.radio("Select Mood:", ["Party Energy", "Deep Focus", "Chill Lofi", "Emotional/Sad", "Devotional"])
        
    mood_map = {
        "Party Energy": "https://open.spotify.com/embed/playlist/37i9dQZF1DXaXB88o2P9G9",
        "Deep Focus": "https://open.spotify.com/embed/playlist/37i9dQZF1DX4sWSp4Status",
        "Chill Lofi": "https://open.spotify.com/embed/playlist/37i9dQZF1DX8UebicO9uaR",
        "Emotional/Sad": "https://open.spotify.com/embed/playlist/37i9dQZF1DX7qK8ma5wgG1",
        "Devotional": "https://open.spotify.com/embed/playlist/37i9dQZF1DX0S69v9S94G0"
    }

    with col_m2:
        st.markdown(f"""
            <div class='glass-card' style='text-align:center;'>
                <h3>Technova AI Recommends: {mood}</h3>
                <iframe src="{mood_map[mood]}" width="100%" height="380" frameborder="0" 
                allowtransparency="true" allow="encrypted-media"></iframe>
            </div>
        """, unsafe_allow_html=True)

elif choice == "🎙️ Creative Studio":
    st.markdown("<h2>Creative Studio</h2>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🎙️ LIVE VOICE", "📤 UPLOAD", "✍️ TEXT-TO-MELODY"])
    
    with tab1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        rec = st.audio_input("Record your voice now:")
        if rec:
            y, sr = librosa.load(rec)
            if st.button("TRANSFORM VOICE"):
                st.balloons()
                out = transform_voice(y, sr)
                st.audio(out, sample_rate=sr)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        up = st.file_uploader("Upload Audio File", type=['mp3', 'wav'])
        if up:
            y, sr = librosa.load(up)
            if st.button("PROCESS FILE"):
                out = transform_voice(y, sr)
                st.audio(out, sample_rate=sr)

    with tab3:
        txt = st.text_input("Type a message to create a melody:")
        if txt and st.button("GENERATE TUNE"):
            st.toast("Generating Technova Tone...")
            freq = (sum([ord(c) for c in txt]) % 400) + 200
            t = np.linspace(0, 4, 44100 * 4)
            melody = 0.5 * np.sin(2 * np.pi * freq * t)
            st.audio(melody, sample_rate=44100)

elif choice == "🌈 Sensory Pulse":
    st.markdown("<h2>Sensory Visualization Room</h2>", unsafe_allow_html=True)
    up_v = st.file_uploader("Upload audio to see vibrations", type=['mp3','wav'])
    if up_v:
        y, sr = librosa.load(up_v)
        fig, ax = plt.subplots(figsize=(12, 4), facecolor='black')
        ax.plot(y[::100], color='#ff00c1')
        ax.set_axis_off()
        st.pyplot(fig)

elif choice == "♿ Accessibility":
    st.markdown("<h2>Inclusive Hearing Assist</h2>", unsafe_allow_html=True)
    st.write("Technova's special frequency shifting for the hearing impaired.")
    up_h = st.file_uploader("Input audio for optimization", type=['mp3', 'wav'])
    if up_h:
        y, sr = librosa.load(up_h)
        shift = st.slider("Pitch Down (Lower = more feelable vibrations)", -15, 0, -8)
        if st.button("OPTIMIZE FOR VIBRATION"):
            st.snow()
            y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift)
            st.audio(y_shift * 1.5, sample_rate=sr)
            st.info("Now connect your Earspots for haptic feedback.")
