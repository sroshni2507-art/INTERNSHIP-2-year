import streamlit as st
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
import io
import time
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="SonicSense Ultra", layout="wide", initial_sidebar_state="expanded")

# --- BEAUTIFUL CUSTOM CSS (Neon Glassmorphism) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
        background-color: #050505;
        color: #e0e0e0;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1e2a4a, #050505);
    }

    /* Card Styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }

    /* Neon Titles */
    h1, h2 {
        background: linear-gradient(to right, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }

    /* Custom Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #00c6ff, #0072ff);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 114, 255, 0.5);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 10, 10, 0.8);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- UTILITY FUNCTIONS ---
def generate_music_from_audio(audio_data, sr):
    f0, _, _ = librosa.pyin(audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = np.nan_to_num(f0)
    hop_length = 512
    f0_stretched = np.repeat(f0, hop_length)[:len(audio_data)]
    phase = np.cumsum(2 * np.pi * f0_stretched / sr)
    music = 0.5 * np.sin(phase) + 0.2 * np.sin(2 * phase)
    music[f0_stretched == 0] = 0
    return music / (np.max(np.abs(music)) + 1e-6)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("## 💎 SonicSense Pro")
    menu = ["🏠 Dashboard", "🎨 Creative Studio", "🧠 Mood & Spotify AI", "🌈 Sensory Room", "🎬 Movie Lab"]
    choice = st.radio("Navigation", menu)
    st.write("---")
    st.write("v2.0 Beta - Inclusive AI")

# --- DASHBOARD ---
if choice == "🏠 Dashboard":
    st.title("Welcome to SonicSense Ultra")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='glass-card'><h3>Accessibility</h3><p>Hearing Impaired Optimized</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='glass-card'><h3>Creative AI</h3><p>Voice to Instrument Hub</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='glass-card'><h3>Productivity</h3><p>Mood-based Workflow</p></div>", unsafe_allow_html=True)
    
    st.image("https://images.unsplash.com/photo-1470225620780-dba8ba36b745?auto=format&fit=crop&q=80&w=1000", use_container_width=True)

# --- CREATIVE STUDIO (MERGED INPUTS) ---
elif choice == "🎨 Creative Studio":
    st.title("Creative Sound Studio")
    st.write("Choose how you want to create music:")
    
    tab1, tab2, tab3 = st.tabs(["🎙️ Record Live", "📁 Upload File", "✍️ Text to Sound"])
    
    with tab1:
        st.subheader("On-the-spot Recording")
        recorded_audio = st.audio_input("Speak or Hum now...")
        if recorded_audio:
            y, sr = librosa.load(recorded_audio)
            if st.button("Convert Live Voice to Music"):
                music = generate_music_from_audio(y, sr)
                st.audio(music, sample_rate=sr)
                st.success("Conversion Complete!")

    with tab2:
        st.subheader("Upload Audio File")
        uploaded_file = st.file_uploader("Choose a wav/mp3 file", type=['wav', 'mp3'])
        if uploaded_file:
            y, sr = librosa.load(uploaded_file)
            if st.button("Process Uploaded Audio"):
                music = generate_music_from_audio(y, sr)
                st.audio(music, sample_rate=sr)

    with tab3:
        st.subheader("Text to Melodic Frequency")
        user_text = st.text_input("Enter a word or sentence (AI will create a tone):")
        if user_text:
            # Creative Logic: Convert text characters to frequencies
            duration = 2.0
            sr = 22050
            t = np.linspace(0, duration, int(sr * duration))
            freq = sum([ord(c) for c in user_text]) % 500 + 200 # Map text to a freq range
            text_music = 0.5 * np.sin(2 * np.pi * freq * t)
            st.write(f"Tone Frequency based on text: {freq}Hz")
            st.audio(text_music, sample_rate=sr)

# --- MOOD AI & SPOTIFY ---
elif choice == "🧠 Mood & Spotify AI":
    st.title("Mood AI Coach")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        u_mood = st.selectbox("Current Mood", ["Energetic", "Calm", "Focused", "Stressed"])
        u_goal = st.text_input("What is your goal?", "Finish my Internship report")
        st.markdown("</div>", unsafe_allow_html=True)
        
    if st.button("Get My Spotify Plan"):
        st.balloons()
        recs = {
            "Energetic": ("Workout", "https://open.spotify.com/search/workout%20edm"),
            "Calm": ("Relax", "https://open.spotify.com/search/lofi%20chill"),
            "Focused": ("Study", "https://open.spotify.com/search/deep%20focus"),
            "Stressed": ("Meditation", "https://open.spotify.com/search/meditation%20piano")
        }
        genre, link = recs[u_mood]
        st.markdown(f"### Activity: {genre} Session")
        st.markdown(f"<a href='{link}' target='_blank'><button style='background:#1DB954; color:white; border:none; padding:15px; border-radius:10px;'>🎧 Open {u_mood} Playlist on Spotify</button></a>", unsafe_allow_html=True)
        
        # Activity Download
        st.download_button("📥 Download My Schedule", f"Mood: {u_mood}\nGoal: {u_goal}\nMusic: {genre}", "schedule.txt")

# --- SENSORY ROOM (ACCESSIBILITY) ---
elif choice == "🌈 Sensory Room":
    st.title("Sensory Sound Experience")
    st.info("Hearing impaired users can visualize and feel the vibrations of audio.")
    
    sens_file = st.file_uploader("Upload Audio for Sensory Map", type=['wav', 'mp3'])
    if sens_file:
        y, sr = librosa.load(sens_file)
        rms = librosa.feature.rms(y=y)[0]
        
        fig, ax = plt.subplots(figsize=(10, 3))
        # Beautiful Gradient Waveform
        ax.plot(rms, color='#00d2ff', alpha=0.7)
        ax.fill_between(range(len(rms)), rms, color='#3a7bd5', alpha=0.3)
        ax.set_axis_off()
        st.pyplot(fig)
        
        st.markdown("### 🫨 Tactile Vibration Simulation")
        if st.button("Simulate Beat Flash"):
            for _ in range(3):
                st.markdown("<div style='height:20px; background:cyan; border-radius:10px; box-shadow: 0 0 20px cyan;'></div>", unsafe_allow_html=True)
                time.sleep(0.2)
                st.markdown("<div style='height:20px; background:magenta; border-radius:10px; box-shadow: 0 0 20px magenta;'></div>", unsafe_allow_html=True)
                time.sleep(0.2)

# --- MOVIE LAB (SUBTITLES) ---
elif choice == "🎬 Movie Lab":
    st.title("Movie Accessibility Lab")
    video = st.file_uploader("Upload Video File", type=['mp4'])
    if video:
        st.video(video)
        if st.button("Generate AI Subtitles"):
            with st.spinner("AI is analyzing speech..."):
                time.sleep(2) # Simulating API call
                st.success("Subtitle Generated: 'Welcome to SonicSense, where sound meets vision.'")
                st.info("Emotion Detected: 😊 Positive / Inspiring")
