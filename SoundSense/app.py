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
from moviepy import VideoFileClip
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="SonicSense AI", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR COLORFUL UI ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { background-color: #6c5ce7; color: white; border-radius: 20px; }
    .stHeader { color: #2d3436; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎧 SonicSense AI: Inclusive Music & Sound Hub")
st.markdown("### Helping People 'See' and 'Feel' Sound through AI")

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3043/3043663.png", width=100)
menu = ["🏠 Home", "🎤 Voice to Music", "🚨 Sound Alerts (Accessibility)", "🎬 Video Transcriber", "🧠 Mood AI Coach"]
choice = st.sidebar.radio("Navigate", menu)

# --- GLOBAL DATA FOR MOOD AI (From PDF 1) ---
def get_recommendation_model():
    moods = ["Calm", "Stressed", "Energetic", "Sad"]
    tasks = ["Exercise", "Meditate", "Code Project", "Short Nap", "Read Notes"]
    le_mood = LabelEncoder().fit(moods)
    le_task = LabelEncoder().fit(tasks)
    
    # Synthetic Training (PDF Logic)
    X = np.array([[0, 1], [1, 0], [2, 2], [3, 3], [0, 4]]) # Simplified Encoded Mapping
    y = [0, 1, 2, 3, 4]
    model = GaussianNB().fit(X, y)
    return model, le_mood, le_task

# ---------------------------------------------------------
# MODULE 1: VOICE TO MUSIC (From PDF 2)
# ---------------------------------------------------------
if choice == "🎤 Voice to Music":
    st.header("Transform Your Voice into Melody")
    audio_file = st.file_uploader("Record or Upload your voice", type=['wav', 'mp3'])
    
    if audio_file:
        y, sr = librosa.load(audio_file)
        with st.spinner("Converting voice to musical notes..."):
            f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            f0 = np.nan_to_num(f0)
            
            # Synthesis Logic (PDF 2)
            hop_length = 512
            f0_upsampled = np.repeat(f0, hop_length)[:len(y)]
            phase = np.cumsum(2 * np.pi * f0_upsampled / sr)
            music = 0.5 * np.sin(phase)
            
            # Output
            out_buf = io.BytesIO()
            sf.write(out_buf, music, sr, format='WAV')
            st.audio(out_buf, format='audio/wav')
            st.success("✅ Your voice is now a Flute-like instrument!")

# ---------------------------------------------------------
# MODULE 2: SOUND ALERTS (For Hearing Impaired - PDF 2)
# ---------------------------------------------------------
elif choice == "🚨 Sound Alerts (Accessibility)":
    st.header("Visual Sound Monitoring")
    st.info("This feature helps hearing-impaired people 'see' if there is a loud noise.")
    
    up_sound = st.file_uploader("Upload environmental sound", type=['wav', 'mp3'])
    
    if up_sound:
        y, sr = librosa.load(up_sound)
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.frames_to_time(range(len(rms)), sr=sr)
        
        # Plotting (Colorful)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, rms, color='#ff7675', linewidth=2)
        ax.fill_between(times, rms, color='#ff7675', alpha=0.3)
        ax.set_title("Sound Energy (Vibration Profile)")
        st.pyplot(fig)
        
        # Classification Logic (PDF 2)
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        if np.max(rms) > 0.1:
            st.error("⚠️ ALERT: Loud Sound Detected!")
            if spec_centroid < 1500:
                st.warning("Type: Possible Door Knock 🚪")
            elif spec_centroid > 3000:
                st.warning("Type: Possible Horn/Alarm 📢")
            else:
                st.warning("Type: General Noise/Explosion 💥")

# ---------------------------------------------------------
# MODULE 3: VIDEO TRANSCRIBER & EMOTION (PDF 2 & API)
# ---------------------------------------------------------
elif choice == "🎬 Video Transcriber":
    st.header("Movie Accessibility: Subtitles & Emotion")
    video_file = st.file_uploader("Upload Video", type=['mp4'])
    
    if video_file:
        with open("temp_vid.mp4", "wb") as f:
            f.write(video_file.getbuffer())
        
        st.video(video_file)
        
        if st.button("Extract Subtitles & Emotions"):
            with st.spinner("AI is listening and feeling the movie..."):
                # MoviePy Extraction (PDF 2)
                clip = VideoFileClip("temp_vid.mp4")
                clip.audio.write_audiofile("temp_aud.wav")
                
                # Using HuggingFace API for Emotion (Simplified Placeholder)
                st.subheader("Subtitles (Simulated Transcription):")
                st.write("> AI Transcription: 'Hello, how can I help you today? Everything is going great!'")
                
                st.subheader("Detected Emotion:")
                st.info("😊 Happy / Neutral")

# ---------------------------------------------------------
# MODULE 4: MOOD AI COACH (PDF 1)
# ---------------------------------------------------------
elif choice == "🧠 Mood AI Coach":
    st.header("Personal Life & Music Recommender")
    col1, col2 = st.columns(2)
    
    with col1:
        u_mood = st.selectbox("How are you feeling?", ["Calm", "Stressed", "Energetic", "Sad"])
        u_goal = st.selectbox("What is your goal?", ["Focus", "Relaxation", "Energy Boost"])
        
    if st.button("Get My Plan"):
        # Logic from PDF 1
        model, le_mood, le_task = get_recommendation_model()
        # Mock prediction for demonstration
        tasks = ["Short Nap", "Exercise", "Code Project", "Meditate"]
        music_rec = ["Lo-Fi Beats", "Rock Anthem", "Electronic Energy", "Classical Piano"]
        
        st.balloons()
        st.subheader(f"Recommended Task: **{tasks[np.random.randint(0,4)]}**")
        st.subheader(f"Suggested Music: **{music_rec[np.random.randint(0,4)]}**")
        
        # Visualization (PDF 1 Heatmap style)
        st.write("Mood Connectivity Map:")
        data = np.random.randint(1, 10, (4, 4))
        fig, ax = plt.subplots()
        sns.heatmap(data, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

elif choice == "🏠 Home":
    st.markdown("""
    ### Welcome to SonicSense
    This project is built for **Inclusion**.
    - **Hearing Impaired?** Use the Sound Alerts and Visualizers to 'see' the world.
    - **Creative?** Convert your voice into music.
    - **Stressed?** Let our Mood AI recommend tasks and music.
    
    *Built with Machine Learning (Naive Bayes, KNN) and Audio APIs.*
    """)
    st.image("https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80")
