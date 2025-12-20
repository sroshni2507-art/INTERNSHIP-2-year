import streamlit as st
import librosa, librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import whisper
from transformers import pipeline

st.set_page_config(page_title="SoundSense", layout="wide")
st.title("🎧 SoundSense – Inclusive AI Music & Movie App")

# -------- SIDEBAR --------
choice = st.sidebar.selectbox(
    "Select Feature",
    [
        "Voice → Music",
        "Music Visualizer",
        "Movie → Subtitles",
        "Sound Alerts"
    ]
)

# -------- MODULE 1 --------
if choice == "Voice → Music":
    st.header("🎤 Voice to Music")

    audio = st.file_uploader("Upload Voice", type=["wav","mp3"])
    if audio:
        y, sr = librosa.load(audio)
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
        f0 = np.nan_to_num(f0)

        music = np.sin(2*np.pi*f0*np.arange(len(f0))/sr)
        music = music / np.max(np.abs(music))

        sf.write("voice_music.wav", music, sr)
        st.audio("voice_music.wav")

# -------- MODULE 2 --------
elif choice == "Music Visualizer":
    st.header("🎶 Music Visualizer")

    music = st.file_uploader("Upload Music", type=["wav","mp3"])
    if music:
        y, sr = librosa.load(music)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        st.write("Tempo (BPM):", tempo)

        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        st.pyplot(fig)

# -------- MODULE 3 --------
elif choice == "Movie → Subtitles":
    st.header("🎬 Movie Subtitles")

    video = st.file_uploader("Upload Movie", type=["mp4","mkv"])
    if video:
        with open("temp.mp4","wb") as f:
            f.write(video.read())

        clip = VideoFileClip("temp.mp4")
        clip.audio.write_audiofile("temp.wav")

        model = whisper.load_model("base")
        result = model.transcribe("temp.wav")

        emo = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base"
        )

        for seg in result["segments"][:5]:
            emotion = emo(seg["text"])[0]["label"]
            st.write(f"⏱ {seg['start']:.2f}s | {emotion}")
            st.write(seg["text"])

# -------- MODULE 4 --------
elif choice == "Sound Alerts":
    st.header("🚨 Sound Alerts")

    sound = st.file_uploader("Upload Sound", type=["wav","mp3"])
    if sound:
        y, sr = librosa.load(sound)
        rms = np.mean(librosa.feature.rms(y=y))

        if rms > 0.05:
            st.error("⚠️ Loud Sound Detected!")
        else:
            st.success("Sound is Normal")
