import streamlit as st
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import speech_recognition as sr
import io
import os

# Compatibility for MoviePy
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

st.set_page_config(page_title="SonicSense AI", layout="wide")
st.title("🎧 SonicSense: Inclusive AI Audio Hub")

# Sidebar - Simplified strings to avoid mismatch
choice = st.sidebar.radio("Select Module:", 
    ["Voice to Music", 
     "Visual Music", 
     "Subtitle Generator"])

# ---------------------------------------------------------
# MODULE 1: VOICE TO MUSIC
# ---------------------------------------------------------
if choice == "Voice to Music":
    st.header("🎤 Voice to Instrument")
    audio_file = st.file_uploader("Upload Voice Recording", type=["wav", "mp3"])
    
    if audio_file:
        with st.spinner("Processing..."):
            y, sr_rate = librosa.load(audio_file, sr=22050)
            f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            f0 = np.nan_to_num(f0)
            
            hop_length = 512
            f0_stretched = np.repeat(f0, hop_length)[:len(y)]
            phase = np.cumsum(2 * np.pi * f0_stretched / sr_rate)
            music = 0.5 * np.sin(phase) + 0.2 * np.sin(2 * phase)
            music[f0_stretched == 0] = 0
            
            out_buf = io.BytesIO()
            sf.write(out_buf, music, sr_rate, format='WAV')
            st.audio(out_buf, format='audio/wav')

# ---------------------------------------------------------
# MODULE 2: VISUAL MUSIC
# ---------------------------------------------------------
elif choice == "Visual Music":
    st.header("🌈 Visualizing Sound")
    music_file = st.file_uploader("Upload Music File", type=["wav", "mp3"])
    
    if music_file:
        y, sr_rate = librosa.load(music_file)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Waveform")
            fig1, ax1 = plt.subplots()
            librosa.display.waveshow(y, sr=sr_rate, ax=ax1)
            st.pyplot(fig1)
        with col2:
            st.subheader("Spectrogram")
            fig2, ax2 = plt.subplots()
            S = librosa.feature.melspectrogram(y=y, sr=sr_rate)
            librosa.display.specshow(librosa.power_to_db(S), ax=ax2)
            st.pyplot(fig2)

# ---------------------------------------------------------
# MODULE 3: SUBTITLE GENERATOR (FIXED)
# ---------------------------------------------------------
elif choice == "Subtitle Generator":
    st.header("🎬 AI Subtitles")
    st.write("Converts speech from Video/Audio into text.")
    
    media_file = st.file_uploader("Upload Video or Audio", type=["mp4", "wav", "mp3"])
    
    if media_file:
        if st.button("Generate Subtitles"):
            with st.spinner("AI is processing audio..."):
                recognizer = sr.Recognizer()
                
                # Step 1: Save the uploaded file temporarily
                temp_input = "temp_input_file"
                with open(temp_input, "wb") as f:
                    f.write(media_file.getbuffer())
                
                try:
                    # Step 2: Extract or Load Audio using Librosa
                    # This ensures the audio is converted to a format SR can read
                    if media_file.name.endswith("mp4"):
                        video = VideoFileClip(temp_input)
                        video.audio.write_audiofile("temp_raw_audio.wav", logger=None)
                        video.close()
                        audio_to_load = "temp_raw_audio.wav"
                    else:
                        audio_to_load = temp_input
                    
                    # Step 3: CONVERSION TO PCM WAV (The Fix)
                    # We load it with librosa and save it as a standard WAV
                    y, sr_load = librosa.load(audio_to_load, sr=16000) # 16kHz is best for Speech AI
                    clean_wav_path = "clean_speech.wav"
                    sf.write(clean_wav_path, y, sr_load, subtype='PCM_16')
                    
                    # Step 4: Run Speech Recognition
                    with sr.AudioFile(clean_wav_path) as source:
                        audio_data = recognizer.record(source)
                        text = recognizer.recognize_google(audio_data)
                        st.subheader("📜 Transcript:")
                        st.success(text)
                
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Check if the audio has clear speaking voices.")
                
                # Step 5: Cleanup files
                for f in [temp_input, "temp_raw_audio.wav", "clean_speech.wav"]:
                    if os.path.exists(f):
                        os.remove(f)
