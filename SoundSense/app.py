import streamlit as st
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import speech_recognition as sr
import io
import os

# Handle MoviePy v2.0+ and older versions
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

# App Configuration
st.set_page_config(page_title="SonicSense AI", layout="wide")
st.title("🎧 SonicSense: Inclusive AI Audio Hub")
st.markdown("### Bridging Sound and Vision for Everyone")

# Sidebar for Navigation
choice = st.sidebar.radio("Go to Module:", 
    ["🎤 Voice → Music (Creative)", 
     "🌈 Visual Music (For Hearing Impaired)", 
     "🎬 AI Subtitle Generator (Accessibility)"])

# ---------------------------------------------------------
# MODULE 1: VOICE TO MUSIC
# ---------------------------------------------------------
if choice == "Voice → Music (Creative)":
    st.header("Transform Your Voice into an Instrument")
    st.write("Upload your voice (humming or speaking) and the AI will convert it into a flute-like melody.")
    
    audio_file = st.file_uploader("Upload Voice Recording", type=["wav", "mp3"])
    
    if audio_file:
        with st.spinner("Converting voice to melody..."):
            y, sr_rate = librosa.load(audio_file, sr=22050)
            # Pitch Extraction
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
            )
            f0 = np.nan_to_num(f0)
            
            # Synthesize Music (Instrument sound)
            hop_length = 512
            f0_stretched = np.repeat(f0, hop_length)[:len(y)]
            phase = np.cumsum(2 * np.pi * f0_stretched / sr_rate)
            # Mix sine wave with harmonics for a richer 'instrument' sound
            music = 0.5 * np.sin(phase) + 0.2 * np.sin(2 * phase)
            music[f0_stretched == 0] = 0 # Mute silent parts
            
            # Normalize and output
            music = music / (np.max(np.abs(music)) + 1e-6)
            out_buf = io.BytesIO()
            sf.write(out_buf, music, sr_rate, format='WAV')
            
            st.audio(out_buf, format='audio/wav')
            st.success("Musical transformation complete!")

# ---------------------------------------------------------
# MODULE 2: VISUAL MUSIC (For Hearing Impaired)
# ---------------------------------------------------------
elif choice == "Visual Music (For Hearing Impaired)":
    st.header("See the Sound")
    st.write("Visualizations allow users to 'feel' the rhythm and energy of music through patterns.")
    
    music_file = st.file_uploader("Upload a Music File", type=["wav", "mp3"])
    
    if music_file:
        y, sr_rate = librosa.load(music_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Rhythm Pattern (Waveform)")
            fig, ax = plt.subplots()
            librosa.display.waveshow(y, sr=sr_rate, ax=ax, color='#1f77b4')
            st.pyplot(fig)
            
        with col2:
            st.subheader("Frequency Energy (Spectrogram)")
            fig, ax = plt.subplots()
            S = librosa.feature.melspectrogram(y=y, sr=sr_rate)
            S_dB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', ax=ax)
            st.pyplot(fig)

# ---------------------------------------------------------
# MODULE 3: AI SUBTITLE GENERATOR
# ---------------------------------------------------------
elif choice == "AI Subtitle Generator (Accessibility)":
    st.header("Movie Subtitles for Accessibility")
    st.write("Upload a video or audio file to automatically generate text subtitles.")
    
    media_file = st.file_uploader("Upload Movie or Audio", type=["mp4", "wav", "mp3"])
    
    if media_file:
        # Save temp file for processing
        temp_name = f"temp_{media_file.name}"
        with open(temp_name, "wb") as f:
            f.write(media_file.getbuffer())
            
        if media_file.name.endswith("mp4"):
            st.video(media_file)
        else:
            st.audio(media_file)
        
        if st.button("Generate Subtitles Now"):
            with st.spinner("AI is transcribing speech..."):
                recognizer = sr.Recognizer()
                
                # Convert Video to Audio if necessary
                audio_path = "temp_audio.wav"
                if media_file.name.endswith("mp4"):
                    video = VideoFileClip(temp_name)
                    video.audio.write_audiofile(audio_path, logger=None)
                    video.close()
                else:
                    audio_path = temp_name
                
                # Speech-to-Text Recognition
                with sr.AudioFile(audio_path) as source:
                    audio_data = recognizer.record(source)
                    try:
                        text = recognizer.recognize_google(audio_data)
                        st.subheader("📜 Generated Transcript:")
                        st.info(text)
                    except Exception as e:
                        st.error("Could not recognize speech. Ensure the audio is clear.")
        
        # Cleanup
        if os.path.exists("temp_audio.wav"): os.remove("temp_audio.wav")
