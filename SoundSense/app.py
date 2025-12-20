import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import io
from gtts import gTTS

# Page Configuration
st.set_page_config(page_title="SoundSense AI", layout="wide", page_icon="🎵")

st.title("🎵 SoundSense: Live Voice & Text to Song")
st.write("Record your own voice or type text to convert it into a synthesized melody.")

# -------- SIDEBAR SETTINGS (For Loudness & Style) --------
st.sidebar.header("🔊 Audio Customization")
vol_boost = st.sidebar.slider("Volume Gain", 1.0, 10.0, 3.0)
wave_type = st.sidebar.selectbox(
    "Synthesizer Style", 
    ["Square (Loudest/EDM)", "Sine (Soft/Flute)", "Triangle (Vintage)"]
)

# Core Logic: Function to convert Audio into a Melody
def process_to_melody(input_audio, boost=3.0, synth_style="Sine (Soft/Flute)"):
    # Load audio from the provided buffer
    y, sr = librosa.load(input_audio, sr=None)
    
    # 1. Extract Pitch (f0)
    hop_length = 512
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
    )
    f0_cleaned = np.nan_to_num(f0)
    
    if np.max(f0_cleaned) == 0:
        return None, None

    # 2. Fix Duration (Stretch frames to match original sample count)
    total_samples = len(f0_cleaned) * hop_length
    f0_upsampled = np.interp(
        np.arange(total_samples), 
        np.arange(0, total_samples, hop_length), 
        f0_cleaned
    )

    # 3. Create Smooth Waveform using Phase Integration
    phase = 2 * np.pi * np.cumsum(f0_upsampled) / sr
    
    if synth_style == "Square (Loudest/EDM)":
        music = np.sign(np.sin(phase)) # Square wave is naturally very loud
    elif synth_style == "Sine (Soft/Flute)":
        music = np.sin(phase)
    else: # Triangle
        music = 2 * np.abs(2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))) - 1

    # 4. Apply Volume Boost and Prevent Distortion
    music = music * boost
    music = np.clip(music, -1.0, 1.0) # Clipping prevents "crackling"
    
    return music, sr

# -------- APP INTERFACE (Tabs) --------
tab1, tab2 = st.tabs(["🎤 Live Recording", "📝 Text-to-Song"])

# TAB 1: LIVE VOICE RECORDING
with tab1:
    st.subheader("Record Your Voice")
    st.info("Click the microphone icon below to record yourself humming or singing.")
    
    # Streamlit's native audio recording component
    recorded_file = st.audio_input("Start Recording")
    
    if recorded_file:
        if st.button("Generate Song from Recording"):
            with st.spinner("Analyzing your voice..."):
                music, sr = process_to_melody(recorded_file, vol_boost, wave_type)
                if music is not None:
                    buffer = io.BytesIO()
                    sf.write(buffer, music, sr, format='WAV')
                    st.success("Successfully generated!")
                    st.audio(buffer)
                    st.download_button("Download Song", buffer, "live_song.wav")
                else:
                    st.error("Could not detect a clear melody. Please try again more loudly.")

# TAB 2: TEXT TO SONG
with tab2:
    st.subheader("Text to Melodic Speech")
    text_input = st.text_input("Enter text to convert into a song:", "Hello, this is my AI generated song.")
    
    if st.button("Generate Song from Text"):
        if text_input:
            with st.spinner("Converting text to speech..."):
                # Convert Text to Voice (English)
                tts = gTTS(text=text_input, lang='en')
                tts_buf = io.BytesIO()
                tts.write_to_fp(tts_buf)
                tts_buf.seek(0)
                
                st.write("AI Voice Preview:")
                st.audio(tts_buf)

            with st.spinner("Synthesizing melody..."):
                # Convert that Voice to Music
                music, sr = process_to_melody(tts_buf, vol_boost, wave_type)
                if music is not None:
                    final_buf = io.BytesIO()
                    sf.write(final_buf, music, sr, format='WAV')
                    st.subheader("Final Song Output")
                    st.audio(final_buf)
                    st.download_button("Download Text-Song", final_buf, "text_song.wav")
                else:
                    st.error("The voice was too flat to extract a melody.")
