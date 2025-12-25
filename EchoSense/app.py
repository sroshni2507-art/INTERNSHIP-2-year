import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from moviepy import VideoFileClip  # New MoviePy Syntax
import whisper
from transformers import pipeline
import tempfile
import os

# 1. Page Configuration
st.set_page_config(page_title="EchoSense AI", layout="wide", page_icon="🎙️")
st.title("🎙️ EchoSense: Audio & Video AI Analyzer")

# 2. Load Models with Cache (Memory Optimized)
@st.cache_resource
def load_ai_models():
    # 'tiny' model is used to prevent RAM crash on free servers
    w_model = whisper.load_model("tiny")
    e_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    return w_model, e_pipe

with st.spinner("AI மாடல்கள் லோடு ஆகிறது... சற்று காத்திருக்கவும்..."):
    whisper_model, emotion_pipe = load_ai_models()

# 3. Sound Classification Function
def classify_sound_type(y, sr):
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    if spec_centroid < 1500 and zcr < 0.1:
        return "Door Knock 🚪"
    elif spec_centroid > 3000:
        return "Horn 🎺"
    else:
        return "Explosion/General Noise 💥"

# 4. Sidebar Upload Section
st.sidebar.header("Media Upload")
uploaded_file = st.sidebar.file_uploader("Upload Audio/Video", type=["mp3", "wav", "mp4", "m4a", "mov"])

if uploaded_file is not None:
    # Save upload to a temp file
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
        tfile.write(uploaded_file.read())
        temp_path = tfile.name

    # Video processing logic
    final_audio_path = temp_path
    if suffix.lower() in [".mp4", ".mov", ".m4a"]:
        st.sidebar.info("🎥 Video/Media detected. Extracting audio...")
        try:
            video = VideoFileClip(temp_path)
            audio_temp = "extracted_audio.wav"
            # Extracting audio and saving as wav
            video.audio.write_audiofile(audio_temp, logger=None)
            final_audio_path = audio_temp
        except Exception as e:
            st.error(f"Error processing video: {e}")

    # Load audio data for analysis (Resample to 16kHz for Whisper compatibility)
    y, sr = librosa.load(final_audio_path, sr=16000)
    st.sidebar.success("✅ File Loaded")
    st.audio(final_audio_path)

    # 5. UI Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visuals", "🧠 AI Insights", "🎼 Melody Gen", "📝 Transcription"])

    with tab1:
        st.subheader("Audio Waveform & Spectrogram")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_title("Waveform")
            st.pyplot(fig)
        with col2:
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax2)
            plt.colorbar(img, ax=ax2)
            ax2.set_title("Spectrogram")
            st.pyplot(fig2)

    with tab2:
        st.subheader("Intelligent Audio Metrics")
        # Tempo detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        energy = np.mean(librosa.feature.rms(y=y))
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated BPM", f"{float(tempo):.1f}")
        c2.metric("Detected Sound", classify_sound_type(y, sr))
        
        # Emotion logic based on tempo
        if float(tempo) > 120 and energy > 0.05: mood = "Energetic 🔥"
        elif float(tempo) < 80: mood = "Sad/Calm 😢"
        else: mood = "Neutral 😌"
        c3.metric("Audio Mood", mood)

    with tab3:
        st.subheader("Pitch-based Melody Synthesis")
        st.write("Extracting voice pitch and converting to sine wave melody...")
        if st.button("Generate Melody"):
            with st.spinner("Processing Pitch..."):
                f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                f0_clean = np.nan_to_num(f0)
                hop_len = 512
                total_s = len(f0_clean) * hop_len
                f0_up = np.interp(np.arange(total_s), np.arange(0, total_s, hop_len), f0_clean)
                phase = 2 * np.pi * np.cumsum(f0_up) / sr
                music = np.sin(phase)
                out_path = "synth_melody.wav"
                sf.write(out_path, music, sr)
                st.audio(out_path)
                st.success("Synthesis Complete!")

    with tab4:
        st.subheader("AI Transcription & Text Emotion")
        if st.button("Start Transcription"):
            with st.spinner("AI is listening..."):
                # Passing the audio array directly to avoid path issues
                result = whisper_model.transcribe(y)
                st.markdown(f"**Full Transcript:** \n\n {result['text']}")
                
                st.markdown("---")
                st.write("**Segment Analysis:**")
                for segment in result['segments'][:10]:
                    text_snippet = segment['text']
                    # Get emotion for each text segment
                    text_emo = emotion_pipe(text_snippet)[0]['label']
                    st.write(f"🕒 {segment['start']:.1f}s | **{text_emo.upper()}**: {text_snippet}")

else:
    st.info("👈 இடதுபுறம் உள்ள Sidebar-ல் ஒரு ஆடியோ அல்லது வீடியோ கோப்பை அப்லோட் செய்யவும்.")
