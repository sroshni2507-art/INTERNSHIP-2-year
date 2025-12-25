import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from moviepy import VideoFileClip  # New MoviePy v2.0+ Syntax
import whisper
from transformers import pipeline
import tempfile
import os

# Page Setup
st.set_page_config(page_title="EchoSense AI", layout="wide")
st.title("🎙️ EchoSense: Audio & Video AI Analyzer")

# --- Load Models (Cached) ---
@st.cache_resource
def load_ai_models():
    # 'tiny' model is used to prevent RAM crash on Streamlit Cloud
    w_model = whisper.load_model("tiny")
    e_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    return w_model, e_pipe

with st.spinner("Loading AI Models... Please wait."):
    whisper_model, emotion_pipe = load_ai_models()

# --- Functions ---
def classify_sound_type(y, sr):
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    if spec_centroid < 1500 and zcr < 0.1:
        return "Door Knock 🚪"
    elif spec_centroid > 3000:
        return "Horn 🎺"
    else:
        return "Explosion/Other Noise 💥"

# --- Sidebar ---
st.sidebar.header("Upload Media")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["mp3", "wav", "mp4", "m4a", "mov"])

if uploaded_file is not None:
    # Save upload to a temp file
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
        tfile.write(uploaded_file.read())
        temp_path = tfile.name

    # If it's a video, extract audio
    final_audio_path = temp_path
    if suffix.lower() in [".mp4", ".mov"]:
        st.sidebar.info("🎥 Video detected. Extracting audio...")
        try:
            video = VideoFileClip(temp_path)
            audio_temp = "extracted_audio.wav"
            video.audio.write_audiofile(audio_temp, logger=None)
            final_audio_path = audio_temp
        except Exception as e:
            st.error(f"Error processing video: {e}")

    # Load Audio for analysis
    y, sr = librosa.load(final_audio_path)
    st.sidebar.success("✅ File Loaded Successfully")
    st.audio(final_audio_path)

    # Tabs for different features
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visuals", "🧠 AI Analysis", "🎼 Pitch Synthesis", "📝 Transcription"])

    with tab1:
        st.subheader("Audio Waveform & Spectrogram")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_title("Waveform")
            st.pyplot(fig)
        with col2:
            fig2, ax2 = plt.subplots()
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax2)
            plt.colorbar(img, ax=ax2)
            ax2.set_title("Spectrogram")
            st.pyplot(fig2)

    with tab2:
        st.subheader("Intelligent Analysis")
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        energy = np.mean(librosa.feature.rms(y=y))
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Tempo (BPM)", f"{float(tempo):.1f}")
        c2.metric("Sound Type", classify_sound_type(y, sr))
        
        emotion = "Energetic 🔥" if float(tempo) > 120 and energy > 0.05 else "Calm 😌"
        if float(tempo) < 80: emotion = "Sad 😢"
        c3.metric("Detected Mood", emotion)

    with tab3:
        st.subheader("Pitch-to-Music Synthesis")
        if st.button("Generate Pitch Melody"):
            with st.spinner("Processing Pitch..."):
                f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                f0_clean = np.nan_to_num(f0)
                hop_len = 512
                total_s = len(f0_clean) * hop_len
                f0_up = np.interp(np.arange(total_s), np.arange(0, total_s, hop_len), f0_clean)
                phase = 2 * np.pi * np.cumsum(f0_up) / sr
                music = np.sin(phase)
                out_path = "synth_voice.wav"
                sf.write(out_path, music, sr)
                st.audio(out_path)
                st.success("Generated sine-wave melody based on your voice pitch!")

    with tab4:
        st.subheader("Speech-to-Text & Emotion")
        if st.button("Start Transcribing"):
            with st.spinner("Converting speech to text..."):
                result = whisper_model.transcribe(final_audio_path)
                st.markdown(f"**Full Transcript:** \n\n {result['text']}")
                
                st.markdown("---")
                st.write("**Segment-wise Emotion:**")
                for segment in result['segments'][:8]: # Show first 8 segments
                    text_emo = emotion_pipe(segment['text'])[0]['label']
                    st.write(f"🕒 {segment['start']:.1f}s - {segment['end']:.1f}s | **{text_emo}**: {segment['text']}")

else:
    st.info("👈 Please upload an audio or video file in the sidebar to start.")
