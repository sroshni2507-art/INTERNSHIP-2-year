import streamlit as st
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import io
from gtts import gTTS

# App Configuration
st.set_page_config(page_title="SoundSense Pro", layout="wide", page_icon="🎧")
st.title("🎧 SoundSense – AI Voice, Text & Music App")

# Helper Function for Synthesis (To avoid repeating code)
def synthesize_melody(y, sr):
    hop_length = 512
    # 1. Pitch Extraction
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr, hop_length=hop_length
    )
    f0_cleaned = np.nan_to_num(f0)
    
    if np.max(f0_cleaned) == 0:
        return None

    # 2. Upsample to match original length (Fixes the 0:00 duration issue)
    total_samples = len(f0_cleaned) * hop_length
    f0_upsampled = np.interp(
        np.arange(total_samples), 
        np.arange(0, total_samples, hop_length), 
        f0_cleaned
    )

    # 3. Smooth Synthesis (Phase Integration)
    phase = 2 * np.pi * np.cumsum(f0_upsampled) / sr
    music = np.sin(phase)
    
    # Normalize volume
    music = music / (np.max(np.abs(music)) + 1e-6)
    return music

# -------- SIDEBAR --------
choice = st.sidebar.selectbox(
    "Select Module",
    [
        "Text → Voice → Music",
        "Voice → Music (Upload)",
        "Music Visualizer",
        "Sound Alerts"
    ]
)

# -------- MODULE 1: Text → Voice → Music --------
if choice == "Text → Voice → Music":
    st.header("📝 Text to Voice to Music")
    st.write("Type something, and we will turn the spoken words into a melody!")
    
    user_text = st.text_area("Enter text here:", "Hello, I am becoming a song.")
    
    if st.button("Generate Song from Text"):
        if user_text:
            with st.spinner("Converting text to speech..."):
                # 1. Text to Speech
                tts = gTTS(text=user_text, lang='en')
                tts_buffer = io.BytesIO()
                tts.write_to_fp(tts_buffer)
                tts_buffer.seek(0)
                
                # 2. Load the generated speech into librosa
                y, sr = librosa.load(tts_buffer)
                st.subheader("1. Generated Voice")
                st.audio(tts_buffer)

            with st.spinner("Converting voice to melody..."):
                # 3. Synthesize melody
                music = synthesize_melody(y, sr)
                
                if music is not None:
                    # 4. Save to buffer
                    out_buffer = io.BytesIO()
                    sf.write(out_buffer, music, sr, format='WAV')
                    out_buffer.seek(0)

                    st.subheader("2. Generated Music Melody")
                    st.audio(out_buffer)
                    st.download_button("Download Song", out_buffer, "text_song.wav", "audio/wav")
                else:
                    st.error("The voice was too robotic or quiet to extract a melody.")

# -------- MODULE 2: Voice → Music (Upload) --------
elif choice == "Voice → Music (Upload)":
    st.header("🎤 Voice to Music (Upload)")
    audio_file = st.file_uploader("Upload your humming/singing", type=["wav","mp3"])
    
    if audio_file:
        y, sr = librosa.load(audio_file)
        st.audio(audio_file)
        
        if st.button("Convert to Music"):
            music = synthesize_melody(y, sr)
            if music is not None:
                out_buffer = io.BytesIO()
                sf.write(out_buffer, music, sr, format='WAV')
                out_buffer.seek(0)
                st.audio(out_buffer)
                st.success("🎶 Melody generated!")
            else:
                st.error("No clear pitch detected.")

# -------- MODULE 3: Music Visualizer --------
elif choice == "Music Visualizer":
    st.header("🎶 Music Visualization")
    music_file = st.file_uploader("Upload Music", type=["wav","mp3"])
    if music_file:
        y, sr = librosa.load(music_file)
        st.audio(music_file)
        
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        energy = np.mean(librosa.feature.rms(y=y))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tempo", f"{int(tempo)} BPM")
            fig, ax = plt.subplots()
            librosa.display.waveshow(y, sr=sr, ax=ax)
            st.pyplot(fig)
        with col2:
            emotion = "Energetic 🔥" if tempo > 120 else "Calm 😊"
            st.subheader(f"Mood: {emotion}")

# -------- MODULE 4: Sound Alerts --------
elif choice == "Sound Alerts":
    st.header("🚨 Sound Event Alerts")
    sound_file = st.file_uploader("Upload Sound", type=["wav","mp3"])
    if sound_file:
        y, sr = librosa.load(sound_file)
        rms = np.mean(librosa.feature.rms(y=y))
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        if rms > 0.05:
            st.error("⚠️ Loud Sound Detected!")
        
        if spec_centroid < 1500:
            st.write("Detected Type: Door Knock 🚪")
        elif spec_centroid > 3000:
            st.write("Detected Type: Horn/Alarm 🚗")
        else:
            st.write("Detected Type: Explosion/Thump 💥")
