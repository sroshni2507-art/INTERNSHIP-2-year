import streamlit as st
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import io
import speech_recognition as sr
from gtts import gTTS

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="SoundSense Pro", layout="wide", page_icon="🎧")
st.title("🎧 SoundSense – AI Music & Voice App")

# ---------------- SIDEBAR ----------------
choice = st.sidebar.selectbox(
    "Select Module",
    [
        "Voice ➔ Lyrics ➔ Song",
        "Voice ➔ Music (Direct)",
        "Music Visualizer",
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

    # Upsample to match original length
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

def stt_convert(audio_file):
    """Converts recorded audio to text using SpeechRecognition."""
    r = sr.Recognizer()
    # Read the audio file from the Streamlit UploadedFile/AudioBuffer
    with sr.AudioFile(audio_file) as source:
        audio_data = r.record(source)
    try:
        return r.recognize_google(audio_data)
    except Exception as e:
        return f"Error: {str(e)}. Please speak clearly."

# ---------------- MODULES ----------------

# 1. Voice ➔ Lyrics ➔ Song
if choice == "Voice ➔ Lyrics ➔ Song":
    st.header("🎤 Voice to Lyrics to Song")
    st.write("Step 1: Record ➔ Step 2: Edit Text ➔ Step 3: Generate Song")
    
    rec_voice = st.audio_input("Record your speech")
    if rec_voice:
        if 'lyrics' not in st.session_state:
            with st.spinner("Converting speech to text..."):
                st.session_state.lyrics = stt_convert(rec_voice)
        
        edited_lyrics = st.text_area("Edit your Lyrics/Text here:", st.session_state.lyrics)
        
        if st.button("Generate Final Song"):
            with st.spinner("Generating AI Voice & Melody..."):
                # Convert text back to AI voice
                tts = gTTS(text=edited_lyrics, lang='en')
                t_buf = io.BytesIO()
                tts.write_to_fp(t_buf)
                t_buf.seek(0)
                
                # Convert AI voice to loud song
                music, sr_out = synthesize_musical_song(t_buf, vol_boost, wave_style)
                if music is not None:
                    res_buf = io.BytesIO()
                    sf.write(res_buf, music, sr_out, format='WAV')
                    st.success("Song Generated!")
                    st.audio(res_buf)
                    st.download_button("Download Song", res_buf, "ai_lyrics_song.wav")

# 2. Voice ➔ Music (Direct Upload)
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

# 4. Sound Alerts
elif choice == "Sound Alerts":
    st.header("🚨 Sound Event Alerts")
    sound_file = st.file_uploader("Upload Sound", type=["wav","mp3"])
    if sound_file:
        y, sr = librosa.load(sound_file)
        rms = np.mean(librosa.feature.rms(y=y))
        if rms > 0.05: st.error("⚠️ Loud Sound Detected!")
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        st.write("Type:", "Horn 🚗" if spec_centroid > 3000 else "Door Knock 🚪")
