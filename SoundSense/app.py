import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import io

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TECHNOVA SOLUTION | SonicSense Pro",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ADVANCED CSS (VISIBILITY & BRANDING) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&family=Poppins:wght@300;400;600;800&display=swap');

    /* Full Page Background */
    .stApp {
        background: url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
    }
    
    /* Overlay for Content Clarity */
    .main-overlay {
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0, 0, 0, 0.75);
        z-index: -1;
    }

    /* --- SIDEBAR CUSTOMIZATION (ULTRA VISIBLE) --- */
    [data-testid="stSidebar"] {
        background-color: #0a0a20 !important; /* Deep Dark Blue */
        border-right: 2px solid #00d2ff !important;
    }

    /* Sidebar Text (Word Colors) */
    [data-testid="stSidebar"] * {
        color: #00d2ff !important; /* Cyan for readability */
        font-family: 'Poppins', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        color: #ff00c1 !important; /* Pink for headers */
        font-family: 'Orbitron', sans-serif !important;
        text-align: center;
    }

    /* TECHNOVA HERO HEADER */
    .hero-container {
        text-align: center;
        padding: 40px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 40px;
        border: 2px solid #00d2ff;
        backdrop-filter: blur(20px);
        margin-bottom: 30px;
        box-shadow: 0 0 50px rgba(0, 210, 255, 0.3);
    }

    .company-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 5.5rem !important;
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

    /* Vibrant Glass Cards */
    .glass-card {
        background: rgba(10, 10, 20, 0.85);
        padding: 30px;
        border-radius: 25px;
        border: 1px solid rgba(0, 210, 255, 0.5);
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 15px 45px rgba(0,0,0,0.7);
    }

    /* Big Fonts for Content */
    h2, h3 { color: #00d2ff !important; font-family: 'Orbitron', sans-serif; font-size: 2.8rem !important; }
    p, label, li { font-size: 1.5rem !important; color: #ffffff !important; font-family: 'Poppins', sans-serif; font-weight: 500; }
    
    /* Neon Styled Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #bc13fe, #00d2ff);
        border: none; color: white !important; border-radius: 50px;
        padding: 18px 50px; font-size: 1.6rem !important;
        font-family: 'Orbitron', sans-serif; width: 100%;
        transition: 0.4s ease-in-out;
        box-shadow: 0 5px 25px rgba(188, 19, 254, 0.4);
    }
    .stButton>button:hover { transform: scale(1.03); box-shadow: 0 0 45px #00d2ff; }

    </style>
    <div class="main-overlay"></div>
    """, unsafe_allow_html=True)

# --- 3. CORE AI LOGIC ---
def voice_to_music_ai(audio, sr):
    f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = np.nan_to_num(f0)
    hop = 512
    f0_stretched = np.repeat(f0, hop)[:len(audio)]
    phase = np.cumsum(2 * np.pi * f0_stretched / sr)
    synth = 0.6 * np.sin(phase) + 0.3 * np.sin(2 * phase)
    return synth / (np.max(np.abs(synth)) + 1e-6)

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h2>TECHNOVA</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=120)
    st.write("---")
    menu = ["🏠 Dashboard", "🧠 Mood & Spotify AI", "🎙️ Creative AI Studio", "♿ Hearing Assist", "🌈 Sensory Pulse"]
    choice = st.selectbox("SELECT MODULE", menu)
    st.write("---")
    st.success("AI Core Status: Online")
    st.info("Version: 5.0 Ultra Pro")

# --- 5. TOP BRANDING HEADER ---
st.markdown("""
    <div class="hero-container">
        <h1 class="company-title">TECHNOVA SOLUTION</h1>
        <p style="letter-spacing: 8px; color:#92fe9d; font-size:1.8rem; font-weight:600;">Next-Gen Audio Intelligence</p>
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
            <p>Welcome to Technova Solution. Use our AI tools to create music, optimize haptic patterns, and boost productivity flow.</p>
            <ul>
                <li>AI Voice-to-Instrument Conversion</li>
                <li>Vibrational Sound Optimization</li>
                <li>Haptic Pattern Visualization</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.image("https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=500&h=500&fit=crop", use_container_width=True)

# --- MOOD & SPOTIFY AI (FIXED) ---
elif choice == "🧠 Mood & Spotify AI":
    st.markdown("<h2>Productivity Flow & Mood AI</h2>", unsafe_allow_html=True)
    
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
        u_goal = st.text_input("YOUR GOAL TODAY:", value="Ex: Finish Internship Project")
        
        if st.button("🚀 LAUNCH SESSION"):
            st.session_state.mood_active = True
            st.balloons() # Balloons
            st.snow()     # Snow
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if 'mood_active' in st.session_state and st.session_state.mood_active:
            playlist_id = mood_spotify_map[u_mood]
            # Fixed embed structure with Generator source
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

# --- CREATIVE AI STUDIO ---
elif choice == "🎙️ Creative AI Studio":
    st.markdown("<h2>AI Creative Studio</h2>", unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["🎙️ LIVE RECORD", "📤 UPLOAD", "✍️ TEXT-TO-MELODY"])
    
    with t1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        v_in = st.audio_input("Record your voice/singing:")
        if v_in:
            y, sr = librosa.load(v_in)
            if st.button("✨ TRANSFORM LIVE VOICE"):
                st.balloons()
                st.snow()
                out = voice_to_music_ai(y, sr)
                st.audio(out, sample_rate=sr)
        st.markdown("</div>", unsafe_allow_html=True)

    with t2:
        up = st.file_uploader("Upload Audio (MP3/WAV)", type=['mp3','wav'])
        if up:
            y, sr = librosa.load(up)
            if st.button("🚀 CONVERT UPLOAD"):
                st.balloons()
                out = voice_to_music_ai(y, sr)
                st.audio(out, sample_rate=sr)

    with t3:
        txt = st.text_input("Type text to create a unique melody:")
        if txt and st.button("🎵 GENERATE TUNE"):
            st.toast("Composing unique tone...")
            f = (sum([ord(c) for c in txt]) % 400) + 200
            t = np.linspace(0, 4, 44100 * 4)
            melody = 0.5 * np.sin(2 * np.pi * f * t)
            st.audio(melody, sample_rate=44100)

# --- HEARING ASSIST ---
elif choice == "♿ Hearing Assist":
    st.markdown("<h2>Inclusive Hearing Lab</h2>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.write("Special frequency shifting for bone-conduction vibrations.")
    up_h = st.file_uploader("Upload audio for haptic optimization", type=['mp3', 'wav'])
    if up_h:
        y, sr = librosa.load(up_h)
        shift = st.slider("Frequency Shift (Lower is more feelable)", -15, 0, -8)
        if st.button("🔊 OPTIMIZE"):
            st.balloons()
            st.snow()
            y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift)
            st.audio(y_shift * 1.8, sample_rate=sr)
            st.success("Optimization Complete. Ready for Earspots.")
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
