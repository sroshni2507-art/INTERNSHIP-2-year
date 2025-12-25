import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import moviepy.editor as mp
import whisper
from transformers import pipeline
import os
import tempfile

# Page Configuration
st.set_page_config(page_title="Audio & Video Analytics AI", layout="wide")
st.title("🎵 Audio & Video Analytics AI Platform")

# --- Load Models (Cached for speed) ---
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

whisper_model = load_whisper_model()
emotion_pipe = load_emotion_model()

# --- Functions ---
def classify_sound_type(y, sr):
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    if spec_centroid < 1500 and zcr < 0.1:
        return "Door Knock 🚪"
    elif spec_centroid > 3000:
        return "Horn 🎺"
    else:
        return "Explosion 💥 / Other"

# --- Sidebar: File Upload ---
st.sidebar.header("Upload Section")
uploaded_file = st.sidebar.file_uploader("Upload Audio or Video", type=["mp3", "wav", "mp4", "m4a"])

if uploaded_file is not None:
    # Create a temporary file to save the upload
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    file_path = tfile.name

    # Check if it's a video file
    if uploaded_file.name.endswith(".mp4"):
        st.info("Extracting audio from video...")
        video = mp.VideoFileClip(file_path)
        audio_path = "temp_audio.wav"
        video.audio.write_audiofile(audio_path, logger=None)
        file_path = audio_path

    # Load Audio
    y, sr = librosa.load(file_path)
    st.success(f"File loaded: {uploaded_file.name}")
    st.audio(file_path)

    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "🧠 AI Analysis", "🎼 Music Synthesis", "📝 Transcription"])

    with tab1:
        st.subheader("Waveform & Spectrogram")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(y, sr=sr, ax=ax)
            plt.title("Voice Waveform")
            st.pyplot(fig)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            spec = librosa.feature.melspectrogram(y=y, sr=sr)
            spec_db = librosa.power_to_db(spec, ref=np.max)
            img = librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax2)
            plt.title("Music Spectrogram")
            plt.colorbar(img, ax=ax2)
            st.pyplot(fig2)

    with tab2:
        st.subheader("BPM & Sound Classification")
        
        # Tempo/BPM
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        st.write(f"**Estimated Tempo (BPM):** {tempo:.2f}")
        
        # Energy and Emotion
        energy = np.mean(librosa.feature.rms(y=y))
        if tempo > 120 and energy > 0.05:
            emotion = "Energetic 🔥"
        elif tempo < 80:
            emotion = "Sad 😢"
        else:
            emotion = "Calm 😌"
        st.write(f"**Detected Music Emotion:** {emotion}")

        # Sound Type Classification
        sound_type = classify_sound_type(y, sr)
        st.write(f"**Detected Sound Type:** {sound_type}")

        # Loud Sound Detection
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.frames_to_time(range(len(rms)), sr=sr)
        threshold = np.mean(rms) + 2 * np.std(rms)
        alert_times = times[rms > threshold]
        
        if len(alert_times) > 0:
            st.warning(f"⚠️ Loud sound detected at {alert_times[0]:.2f}s")

    with tab3:
        st.subheader("Pitch-based Music Synthesis")
        st.write("Extracting pitch (f0) and generating a sine wave...")
        
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_cleaned = np.nan_to_num(f0)
        
        hop_length = 512
        total_samples = len(f0_cleaned) * hop_length
        f0_upsampled = np.interp(np.arange(total_samples), np.arange(0, total_samples, hop_length), f0_cleaned)
        
        phase = 2 * np.pi * np.cumsum(f0_upsampled) / sr
        music = np.sin(phase)
        
        synth_file = "voice_music.wav"
        sf.write(synth_file, music, sr)
        st.write("Generated Music based on your voice:")
        st.audio(synth_file)

    with tab4:
        st.subheader("Speech to Text & Text Emotion")
        if st.button("Start Transcription"):
            with st.spinner("Transcribing..."):
                result = whisper_model.transcribe(file_path)
                text = result['text']
                st.write("**Full Text:**")
                st.info(text)
                
                # Text Emotion Detection
                st.write("**Segment Analysis:**")
                for seg in result['segments'][:5]: # Showing first 5 segments
                    emo_label = emotion_pipe(seg['text'])[0]['label']
                    st.write(f"[{seg['start']:.2f}s] **{emo_label}**: {seg['text']}")

else:
    st.info("Please upload an audio or video file from the sidebar to begin analysis.")
