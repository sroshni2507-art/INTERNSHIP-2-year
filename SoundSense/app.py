import streamlit as st
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import io

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TECHNOVA SOLUTION | SonicSense Pro", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. ADVANCED NEON & GLASSMORPHISM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&family=Inter:wght@300;600;800&display=swap');
    
    /* Background Global Style */
    .stApp {
        background: radial-gradient(circle at top right, #0d0e22, #05060a);
        color: #ffffff;
    }

    /* TECHNOVA PREMIUM HEADER DESIGN */
    .technova-container {
        text-align: center;
        padding: 40px 10px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 30px;
        border: 1px solid rgba(0, 210, 255, 0.3);
        margin-bottom: 40px;
        box-shadow: 0 10px 50px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(10px);
    }

    .company-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 4.8rem !important;
        font-weight: 900;
        letter-spacing: 10px;
        background: linear-gradient(90deg, #00d2ff, #92fe9d, #00d2ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 4s linear infinite;
        margin: 0;
    }

    @keyframes shine {
        to { background-position: 200% center; }
    }

    .company-tagline {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #92fe9d;
        letter-spacing: 6px;
        text-transform: uppercase;
        margin-top: 10px;
        font-weight: 600;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(0, 210, 255, 0.2);
        backdrop-filter: blur(15px);
        margin-bottom: 25px;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #00d2ff, #3a7bd5);
        color: white !important;
        border-radius: 50px;
        padding: 12px 40px;
        font-weight: 800;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(0, 210, 255, 0.6);
        transform: scale(1.02);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #00d2ff33;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CORE LOGIC FUNCTIONS ---
def generate_music_from_audio(audio_data, sr):
    f0, _, _ = librosa.pyin(audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = np.nan_to_num(f0)
    hop_length = 512
    f0_stretched = np.repeat(f0, hop_length)[:len(audio_data)]
    phase = np.cumsum(2 * np.pi * f0_stretched / sr)
    music = 0.5 * np.sin(phase) + 0.2 * np.sin(2 * phase)
    music[f0_stretched == 0] = 0
    if np.max(np.abs(music)) > 0:
        music = music / (np.max(np.abs(music)) + 1e-6)
    return music

# --- 4. SIDEBAR MENU ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; font-family:Orbitron; color:#00d2ff;'>TECHNOVA</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=80)
    st.write("---")
    menu = ["🏠 Dashboard", "👂 Inclusive Hearing Lab", "🎨 Creative Studio", "🧠 Mood AI", "🌈 Sensory Room"]
    choice = st.sidebar.radio("CHOOSE MODULE", menu)
    st.write("---")
    st.caption("© 2025 Technova Solution")

# --- 5. MAIN TOP BRANDING (TECHNOVA HEADER) ---
st.markdown("""
    <div class="technova-container">
        <h1 class="company-title">TECHNOVA SOLUTION</h1>
        <p class="company-tagline">Redefining Sound Through AI</p>
    </div>
    """, unsafe_allow_html=True)

# --- 6. MODULES LOGIC ---

# 1. DASHBOARD
if choice == "🏠 Dashboard":
    st.title("Inclusive Audio AI Hub")
    st.image("https://images.unsplash.com/photo-1550745165-9bc0b252726f?q=80&w=2070&auto=format&fit=crop", use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='glass-card'><h3>Hearing Assist</h3><p>Optimized for hearing impairments & vibrations.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='glass-card'><h3>Creative AI</h3><p>Voice-to-Instrument synthesis at your fingertips.</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='glass-card'><h3>Sensory Art</h3><p>Visualize sound pulses in high definition.</p></div>", unsafe_allow_html=True)

# 2. INCLUSIVE HEARING LAB
elif choice == "👂 Inclusive Hearing Lab":
    st.title("Technova Hearing Assistance Lab")
    st.write("Frequency shifting for those with hearing difficulties. Optimized for bone conduction devices.")
    
    uploaded_file = st.file_uploader("Upload Audio", type=['wav', 'mp3'])
    if uploaded_file:
        y, sr = librosa.load(uploaded_file)
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Hearing Optimization Settings")
        shift_steps = st.slider("Frequency Shift (Downwards)", -12, 0, -5)
        vibe_boost = st.slider("Bass/Vibration Gain", 1.0, 3.0, 1.5)
        
        if st.button("🔊 OPTIMIZE FOR VIBRATIONS"):
            with st.spinner("Processing via Technova AI..."):
                y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift_steps)
                y_final = np.clip(y_shifted * vibe_boost, -1.0, 1.0)
                
                st.success("Optimized for Earspots / Bone Conduction Headphones!")
                st.audio(y_final, sample_rate=sr)
                
                # Visual Feedback
                rms = librosa.feature.rms(y=y_final)[0]
                fig, ax = plt.subplots(figsize=(10, 2), facecolor='#05060a')
                ax.fill_between(range(len(rms)), rms, color='#00d2ff', alpha=0.6)
                ax.set_axis_off()
                st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

# 3. CREATIVE STUDIO
elif choice == "🎨 Creative Studio":
    st.title("Technova AI Sound Studio")
    recorded_audio = st.audio_input("Record Voice to Convert")
    if recorded_audio:
        y, sr_load = librosa.load(recorded_audio)
        if st.button("CONVERT VOICE TO MUSIC"):
            music = generate_music_from_audio(y, sr_load)
            st.audio(music, sample_rate=sr_load)

# 4. MOOD AI
elif choice == "🧠 Mood AI":
    st.title("Focus & Flow State")
    u_mood = st.selectbox("Current Mood", ["Energetic", "Calm", "Focused", "Stressed"])
    if st.button("OPEN SPOTIFY SESSION"):
        st.markdown(f"<a href='https://open.spotify.com' target='_blank'><button style='width:100%;'>🎧 OPEN SPOTIFY FOR {u_mood.upper()}</button></a>", unsafe_allow_html=True)

# 5. SENSORY ROOM
elif choice == "🌈 Sensory Room":
    st.title("Sensory Visualization")
    sens_file = st.file_uploader("Upload sound to 'See' vibrations", type=['wav', 'mp3'])
    if sens_file:
        y, sr_rate = librosa.load(sens_file)
        rms = librosa.feature.rms(y=y)[0]
        fig, ax = plt.subplots(figsize=(10, 3), facecolor='#05060a')
        ax.plot(rms, color='#00d2ff', linewidth=3)
        ax.fill_between(range(len(rms)), rms, color='#3a7bd5', alpha=0.3)
        ax.set_axis_off()
        st.pyplot(fig)
