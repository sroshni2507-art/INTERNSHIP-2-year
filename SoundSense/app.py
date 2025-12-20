import streamlit as st
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import io
import speech_recognition as sr
from gtts import gTTS
from moviepy.editor import VideoFileClip
import whisper
from transformers import pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="SoundSense Pro", layout="wide", page_icon="🎧")
st.title("🎧 SoundSense – Inclusive AI Music & Movie App")

# ---------------- SIDEBAR ----------------
choice = st.sidebar.selectbox(
    "Select Module",
    [
        "Voice ➔ Lyrics ➔ Song (New!)",
        "Voice ➔ Music (Direct)",
        "Music Visualizer",
        "Movie ➔ Subtitles",
        "Sound Alerts"
    ]
)

# Sidebar Settings for Audio Quality
st.sidebar.divider()
st.sidebar.header("🔊 Song Settings")
vol_boost = st.sidebar.slider("Volume Boost", 1.0, 10.0, 5.0)
wave_style = st.sidebar.selectbox("Synth Style", ["Square (Loud)", "Sine (Soft)", "Triangle"])

# ---------------- HELPER FUNCTIONS ----------------

def synthesize_musical_song(audio_input, boost=5.0, style="Square (Loud)"):
    """Advanced synthesis: Corrects duration, pitch, and adds volume envelope."""
    y, sample_rate = librosa.load(audio_input, sr=None)
    hop_length = 512
    # Pitch extraction
    f0, _, _ = librosa.pyin(y, fmin=80, fmax=800, sr=sample_rate)
    f0_cleaned = np.nan_to_num(f0)
    
    # Extract Volume Envelope (To prevent constant buzzing)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms = np.interp(np.arange(len(f0_cleaned)), np.arange(len(rms)), rms)
    rms = rms / (np.max(rms) + 1e-6)

    # Upsample
    total_samples = len(f0_cleaned) * hop_length
    f0_up = np.interp(np.arange(total_samples), np.arange(0, total_samples, hop_length), f0_cleaned)
    rms_up = np.interp(np.arange(total_samples), np.arange(0, total_samples, hop_length), rms)

    # Generate Audio
    phase = 2 * np.pi * np.cumsum(f0_up) / sample_rate
    if style == "Square (Loud)":
        music = np.sign(np.sin(phase))
    elif style == "Sine (Soft)":
        music = np.sin(phase)
    else:
        music = 2 * np.abs(2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))) - 1

    # Apply Volume Envelope & Boost
    music = music * rms_up * boost
    music = np.clip(music, -1.0, 1.0)
    return music, sample_rate

def stt_convert(audio_bytes):
    """Converts Audio bytes to Text."""
    r = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(audio_bytes.read())) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio)
    except:
        return "Could not understand audio."

# ---------------- MODULES ----------------

# 1. NEW: Voice ➔ Lyrics ➔ Song
if choice == "Voice ➔ Lyrics ➔ Song (New!)":
    st.header("🎤 Voice to Lyrics to Song")
    st.write("Record your voice ➔ Edit the text into lyrics ➔ Create a song!")
    
    rec_voice = st.audio_input("Record your speech")
    if rec_voice:
        if 'lyrics' not in st.session_state:
            with st.spinner("Converting speech to text..."):
                st.session_state.lyrics = stt_convert(rec_voice)
        
        edited_lyrics = st.text_area("Refine your Lyrics:", st.session_state.lyrics)
        
        if st.button("Generate Final Song"):
            with st.spinner("Generating AI Voice & Melody..."):
                tts = gTTS(text=edited_lyrics, lang='en')
                t_buf = io.BytesIO()
                tts.write_to_fp(t_buf)
                t_buf.seek(0)
                
                music, sr_out = synthesize_music = synthesize_musical_song(t_buf, vol_boost, wave_style)
                if music is not None:
                    res_buf = io.BytesIO()
                    sf.write(res_buf, music, sr_out, format='WAV')
                    st.audio(res_buf)
                    st.download_button("Download Song", res_buf, "ai_lyrics_song.wav")

# 2. Voice ➔ Music (Original Refined)
elif choice == "Voice ➔ Music (Direct)":
    st.header("🎤 Direct Voice to Music")
    audio_file = st.file_uploader("Upload your voice (.wav/.mp3)", type=["wav","mp3"])
    
    if audio_file:
        st.audio(audio_file)
        if st.button("Generate Music"):
            with st.spinner("Processing..."):
                music, sr_out = synthesize_musical_song(audio_file, vol_boost, wave_style)
                out_buf = io.BytesIO()
                sf.write(out_buf, music, sr_out, format='WAV')
                st.audio(out_buf)

# 3. Music Visualizer
elif choice == "Music Visualizer":
    st.header("🎶 Music Visualization")
    music_file = st.file_uploader("Upload Music (.wav/.mp3)", type=["wav","mp3"])
    if music_file:
        y, sr = librosa.load(music_file)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        st.metric("Tempo (BPM)", round(float(tempo), 2))
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        st.pyplot(fig)
        energy = np.mean(librosa.feature.rms(y=y))
        st.write("Emotion:", "Energetic 🔥" if tempo > 120 else "Calm 😊")

# 4. Movie Subtitles
elif choice == "Movie ➔ Subtitles":
    st.header("🎬 Movie → Subtitles + Emotion")
    video_file = st.file_uploader("Upload Movie (.mp4)", type=["mp4"])
    if video_file:
        with open("temp_vid.mp4","wb") as f: f.write(video_file.read())
        clip = VideoFileClip("temp_vid.mp4")
        clip.audio.write_audiofile("temp_aud.wav")
        
        st.info("🎙 Transcribing (Whisper)...")
        model = whisper.load_model("base")
        result = model.transcribe("temp_aud.wav")
        
        emo = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
        for seg in result["segments"][:5]:
            label = emo(seg["text"])[0]["label"]
            st.write(f"⏱ {seg['start']:.2f}s | **{label}**: {seg['text']}")

# 5. Sound Alerts
elif choice == "Sound Alerts":
    st.header("🚨 Sound Event Alerts")
    sound_file = st.file_uploader("Upload Sound", type=["wav","mp3"])
    if sound_file:
        y, sr = librosa.load(sound_file)
        rms = np.mean(librosa.feature.rms(y=y))
        if rms > 0.05: st.error("⚠️ Loud Sound Detected!")
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        st.write("Type:", "Horn 🚗" if spec_centroid > 3000 else "Door Knock 🚪")
