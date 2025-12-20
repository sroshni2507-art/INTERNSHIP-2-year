import streamlit as st
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import io
import os

# Page Config
st.set_page_config(page_title="SonicSense AI", layout="wide")
st.title("🌟 SonicSense: AI for Everyone")
st.markdown("### Creating an Inclusive World through Audio & Vision")

# Sidebar Navigation
choice = st.sidebar.radio("Navigate Modules", 
    ["Voice → Music (Creative)", 
     "Visual Music (For Hearing Impaired)", 
     "Movie Subtitle Generator (Accessibility)"])

# ---------------------------------------------------------
# MODULE 1: VOICE TO MUSIC (ML PITCH TRACKING)
# ---------------------------------------------------------
if choice == "Voice → Music (Creative)":
    st.header("🎤 Voice to Instrument")
    st.write("Convert your hum or speech into a beautiful flute-like melody.")
    
    audio_file = st.file_uploader("Upload or Record Voice", type=["wav", "mp3"])
    
    if audio_file:
        y, sr_rate = librosa.load(audio_file)
        with st.spinner("Analyzing Pitch..."):
            f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            f0 = np.nan_to_num(f0)
            
            # Synthesize Music
            hop_length = 512
            f0_stretched = np.repeat(f0, hop_length)[:len(y)]
            phase = np.cumsum(2 * np.pi * f0_stretched / sr_rate)
            music = 0.6 * np.sin(phase) + 0.2 * np.sin(2 * phase) # Harmonics
            music[f0_stretched == 0] = 0
            
            # Save and Play
            out_buf = io.BytesIO()
            sf.write(out_buf, music, sr_rate, format='WAV')
            st.audio(out_buf, format='audio/wav')
            st.success("Your voice is now a melody!")

# ---------------------------------------------------------
# MODULE 2: VISUAL MUSIC (SENSORY EXPERIENCE)
# ---------------------------------------------------------
elif choice == "Visual Music (For Hearing Impaired)":
    st.header("🌈 Visualizing Sound")
    st.write("People who cannot hear can 'see' the music through these patterns.")
    
    music_file = st.file_uploader("Upload a Song", type=["wav", "mp3"])
    
    if music_file:
        y, sr_rate = librosa.load(music_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Vibration Pattern (Waveform)")
            fig, ax = plt.subplots()
            librosa.display.waveshow(y, sr=sr_rate, ax=ax, color='cyan')
            st.pyplot(fig)
            
        with col2:
            st.subheader("Mood & Energy (Spectrogram)")
            fig, ax = plt.subplots()
            S = librosa.feature.melspectrogram(y=y, sr=sr_rate)
            S_dB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', ax=ax)
            st.pyplot(fig)
            
        st.info("💡 Tip: Fast patterns mean high energy/happy music. Slow patterns mean calm music.")

# ---------------------------------------------------------
# MODULE 3: SUBTITLE GENERATOR (AI SPEECH TO TEXT)
# ---------------------------------------------------------
elif choice == "Movie Subtitle Generator (Accessibility)":
    st.header("🎬 Movie Subtitles & Accessibility")
    st.write("Upload a video/audio to see real-time subtitles for the hearing impaired.")
    
    video_file = st.file_uploader("Upload Video/Audio", type=["mp4", "wav", "mp3"])
    
    if video_file:
        # Save temp file
        with open("temp_video", "wb") as f:
            f.write(video_file.getbuffer())
            
        st.video(video_file)
        
        if st.button("Generate Subtitles"):
            with st.spinner("AI is listening..."):
                recognizer = sr.Recognizer()
                
                # If video, extract audio
                if video_file.name.endswith("mp4"):
                    video = VideoFileClip("temp_video")
                    video.audio.write_audiofile("temp_audio.wav")
                    audio_path = "temp_audio.wav"
                else:
                    audio_path = "temp_video"
                
                # Speech to Text API
                with sr.AudioFile(audio_path) as source:
                    audio_data = recognizer.record(source)
                    try:
                        text = recognizer.recognize_google(audio_data)
                        st.subheader("📜 Generated Subtitles:")
                        st.write(f"**{text}**")
                    except:
                        st.error("Sorry, could not understand the audio.")
