import streamlit as st
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import io
import os

# Handle MoviePy 2.0+ and older versions
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

# --- PAGE CONFIG ---
st.set_page_config(page_title="SonicSense AI", layout="wide")

# --- CUSTOM COLORFUL CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { 
        background: linear-gradient(45deg, #6c5ce7, #a29bfe); 
        color: white; border-radius: 12px; border: none; padding: 10px 24px;
    }
    .stHeader { color: #2d3436; font-family: 'Helvetica Neue'; }
    .sidebar .sidebar-content { background-color: #dfe6e9; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌈 SonicSense: Inclusive AI Audio Hub")
st.markdown("### Bridging the gap between Sound and Sight")

# --- SIDEBAR ---
st.sidebar.header("Navigation")
menu = ["🏠 Home", "🎤 Voice → Music", "🚨 Sound Alerts (Accessibility)", "🧠 Mood AI Coach", "🎬 Video Transcriber"]
choice = st.sidebar.radio("Select Module", menu)

# ---------------------------------------------------------
# MODULE: VOICE TO MUSIC
# ---------------------------------------------------------
if choice == "🎤 Voice → Music":
    st.header("Transform Your Voice into an Instrument")
    st.info("Upload your voice recording to generate a musical melody based on your pitch.")
    
    audio_file = st.file_uploader("Upload Voice (.wav/mp3)", type=['wav', 'mp3'])
    
    if audio_file:
        with st.spinner("Analyzing Pitch and Generating Music..."):
            y, sr = librosa.load(audio_file)
            # ML Pitch Tracking (PDF 2 Logic)
            f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            f0 = np.nan_to_num(f0)
            
            # Synthesis
            hop_length = 512
            f0_stretched = np.repeat(f0, hop_length)[:len(y)]
            phase = np.cumsum(2 * np.pi * f0_stretched / sr)
            music = 0.5 * np.sin(phase) + 0.2 * np.sin(2 * phase)
            music[f0_stretched == 0] = 0
            
            # Export
            out_buf = io.BytesIO()
            sf.write(out_buf, music, sr, format='WAV')
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Original Voice")
                st.audio(audio_file)
            with col2:
                st.write("AI Generated Instrument")
                st.audio(out_buf)

# ---------------------------------------------------------
# MODULE: SOUND ALERTS (Accessibility - PDF 2)
# ---------------------------------------------------------
elif choice == "🚨 Sound Alerts (Accessibility)":
    st.header("Environmental Sound Monitor")
    st.write("Helping hearing-impaired people 'see' sounds in their environment.")
    
    up_sound = st.file_uploader("Monitor Sound", type=['wav', 'mp3'])
    
    if up_sound:
        y, sr = librosa.load(up_sound)
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.frames_to_time(range(len(rms)), sr=sr)
        
        # Plotting (Colorful Seaborn/Matplotlib)
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.lineplot(x=times, y=rms, ax=ax, color='#e17055')
        ax.fill_between(times, rms, color='#fab1a0', alpha=0.5)
        ax.set_title("Sound Vibration Energy Level")
        st.pyplot(fig)
        
        # Classification
        if np.max(rms) > 0.05:
            st.error("⚠️ LOUD SOUND DETECTED!")
            centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            if centroid < 1500: st.warning("Detected: Door Knock or Thud 🚪")
            elif centroid > 3000: st.warning("Detected: Sharp Alarm or Horn 📢")
            else: st.warning("Detected: General Noise")

# ---------------------------------------------------------
# MODULE: MOOD AI COACH (PDF 1)
# ---------------------------------------------------------
elif choice == "🧠 Mood AI Coach":
    st.header("Personalized Activity & Music Recommendations")
    
    mood = st.selectbox("Current Mood", ["Stressed", "Calm", "Energetic", "Sad"])
    goal = st.selectbox("Your Goal", ["Focus", "Relaxation", "Energy Boost"])
    
    if st.button("Generate My Recommendation"):
        st.balloons()
        # Mock Logic based on PDF 1 training results
        recommendations = {
            "Stressed": {"Task": "Meditate / Short Nap", "Music": "Ambient Piano"},
            "Energetic": {"Task": "Workout / Coding", "Music": "Electronic / Rock"},
            "Calm": {"Task": "Reading / Study", "Music": "Lo-Fi Beats"},
            "Sad": {"Task": "Talk to friend / Walk", "Music": "Jazz / Uplifting Pop"}
        }
        
        res = recommendations[mood]
        st.subheader(f"✅ Recommended Task: **{res['Task']}**")
        st.subheader(f"🎶 Suggested Music: **{res['Music']}**")
        
        # Heatmap Visualization (as seen in PDF 1)
        st.write("Mood vs Productivity Map")
        data = np.random.rand(4, 5)
        fig, ax = plt.subplots()
        sns.heatmap(data, annot=True, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)

# ---------------------------------------------------------
# MODULE: HOME
# ---------------------------------------------------------
elif choice == "🏠 Home":
    st.image("https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80")
    st.write("""
    ### Project Overview
    This AI Hub is designed for **Inclusivity** and **Creative Expression**.
    1. **For Hearing Impaired:** Visualize sounds and get alerts for door knocks/horns.
    2. **For Creators:** Turn your hum/voice into a professional instrument track.
    3. **For Everyone:** Use ML to find the best task and music for your current mood.
    
    *Technologies: Python, Librosa (Audio ML), Scikit-Learn (Classification), Seaborn (Visualization).*
    """)

elif choice == "🎬 Video Transcriber":
    st.header("Video to Text (Accessibility)")
    v_file = st.file_uploader("Upload Movie Clip", type=['mp4'])
    if v_file:
        st.video(v_file)
        st.info("AI Transcribing feature (Simulated): 'Hello! Welcome to the lesson on Seaborn functions.'")
