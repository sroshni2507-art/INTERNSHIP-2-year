import streamlit as st
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import whisper
from transformers import pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="SoundSense", layout="wide")
st.title("🎧 SoundSense – Inclusive AI Music & Movie App")

# ---------------- SIDEBAR ----------------
choice = st.sidebar.selectbox(
    "Select Module",
    [
        "Voice → Music",
        "Music Visualizer",
        "Movie → Subtitles",
        "Sound Alerts"
    ]
)

# ---------------- MODULE 1: Voice → Music ----------------
if choice == "Voice → Music":
    st.header("🎤 Voice to Music")
    audio_file = st.file_uploader("Upload your voice (.wav/.mp3)", type=["wav","mp3"])
    
    if audio_file:
        y, sr = librosa.load(audio_file)
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0 = np.nan_to_num(f0)

        music = np.sin(2 * np.pi * f0 * np.arange(len(f0)) / sr)
        music = music / np.max(np.abs(music))

        sf.write("voice_music.wav", music, sr)
        st.audio("voice_music.wav")
        st.success("🎶 Music generated from your voice!")

# ---------------- MODULE 2: Music Visualizer ----------------
elif choice == "Music Visualizer":
    st.header("🎶 Music Visualization")
    music_file = st.file_uploader("Upload Music (.wav/.mp3)", type=["wav","mp3"])
    
    if music_file:
        y, sr = librosa.load(music_file)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        st.write("Tempo (BPM):", tempo)

        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        st.pyplot(fig)

        # RMS energy
        energy = np.mean(librosa.feature.rms(y=y))
        if tempo > 120 and energy > 0.05:
            emotion = "Energetic 🔥"
        elif tempo < 80:
            emotion = "Sad 😢"
        else:
            emotion = "Calm 😊"
        st.write("Detected Music Emotion:", emotion)

# ---------------- MODULE 3: Movie Subtitles ----------------
elif choice == "Movie → Subtitles":
    st.header("🎬 Movie → Subtitles + Emotion")
    video_file = st.file_uploader("Upload Movie (.mp4/.mkv)", type=["mp4","mkv"])
    
    if video_file:
        with open("temp_video.mp4","wb") as f:
            f.write(video_file.read())

        clip = VideoFileClip("temp_video.mp4")
        clip.audio.write_audiofile("temp_audio.wav")

        st.info("🎙 Transcribing audio...")
        model = whisper.load_model("base")
        result = model.transcribe("temp_audio.wav")

        st.info("🎭 Detecting emotions...")
        emo = pipeline("text-classification",
                       model="j-hartmann/emotion-english-distilroberta-base")

        for seg in result["segments"][:5]:
            emotion_label = emo(seg["text"])[0]["label"]
            st.write(f"⏱ {seg['start']:.2f}s | {emotion_label}")
            st.write(seg["text"])

# ---------------- MODULE 4: Sound Alerts ----------------
elif choice == "Sound Alerts":
    st.header("🚨 Sound Event Alerts")
    sound_file = st.file_uploader("Upload Sound (.wav/.mp3)", type=["wav","mp3"])
    
    if sound_file:
        y, sr = librosa.load(sound_file)
        rms = np.mean(librosa.feature.rms(y=y))

        threshold = 0.05
        if rms > threshold:
            st.error("⚠️ Loud Sound Detected!")
        else:
            st.success("Sound is Normal")

        # Simple sound type logic
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        if spec_centroid < 1500 and zcr < 0.1:
            sound_type = "Door Knock 🚪"
        elif spec_centroid > 3000:
            sound_type = "Horn 🚗"
        else:
            sound_type = "Explosion 💥"

        st.write("Detected Sound Type:", sound_type)
