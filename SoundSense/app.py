import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import io
from gtts import gTTS

# App Branding
st.set_page_config(page_title="SoundSense AI Pro", layout="wide", page_icon="🎵")
st.title("🎵 SoundSense: All-in-One AI Song Generator")
st.write("Convert your Live Voice, Uploaded Files, or Text into synthesized music.")

# -------- SIDEBAR: CUSTOMIZATION --------
st.sidebar.header("🔊 Audio & Volume Settings")
vol_boost = st.sidebar.slider("Volume Level (Boost)", 1.0, 10.0, 4.0)
synth_style = st.sidebar.selectbox(
    "Synthesizer Style", 
    ["Square (Loud/Electronic)", "Sine (Soft/Whistle)", "Triangle (Smooth)"]
)

# -------- CORE ENGINE: VOICE TO MUSIC --------
def synthesize_music(audio_input, boost=4.0, style="Square (Loud/Electronic)"):
    # Load audio
    y, sr = librosa.load(audio_input, sr=None)
    
    # 1. Extract Pitch (f0)
    hop_length = 512
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
    )
    f0_cleaned = np.nan_to_num(f0)
    
    if np.max(f0_cleaned) == 0:
        return None, None

    # 2. Fix Duration (Match frames to samples)
    total_samples = len(f0_cleaned) * hop_length
    f0_upsampled = np.interp(
        np.arange(total_samples), 
        np.arange(0, total_samples, hop_length), 
        f0_cleaned
    )

    # 3. Create Waveform
    phase = 2 * np.pi * np.cumsum(f0_upsampled) / sr
    
    if style == "Square (Loud/Electronic)":
        music = np.sign(np.sin(phase)) # Very loud energy
    elif style == "Sine (Soft/Whistle)":
        music = np.sin(phase)
    else: # Triangle
        music = 2 * np.abs(2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))) - 1

    # 4. Boost Volume & Prevent Distortion
    music = music * boost
    music = np.clip(music, -1.0, 1.0) 
    
    return music, sr

# -------- USER INTERFACE TABS --------
tab1, tab2, tab3 = st.tabs(["🎤 Live Recording", "📤 Upload Audio", "📝 Text to Song"])

# TAB 1: LIVE RECORDING
with tab1:
    st.subheader("Record Live Voice")
    recorded_audio = st.audio_input("Click the mic to record")
    
    if recorded_audio:
        if st.button("Convert Recording to Song"):
            with st.spinner("Generating melody..."):
                music, sr = synthesize_music(recorded_audio, vol_boost, synth_style)
                if music is not None:
                    buf = io.BytesIO()
                    sf.write(buf, music, sr, format='WAV')
                    st.audio(buf)
                    st.download_button("Download Song", buf, "live_record_song.wav")
                else:
                    st.error("Could not detect pitch. Try singing more clearly.")

# TAB 2: UPLOAD OPTION
with tab2:
    st.subheader("Upload Audio File")
    uploaded_file = st.file_uploader("Choose a .wav or .mp3 file", type=["wav", "mp3"])
    
    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("Convert Upload to Song"):
            with st.spinner("Processing file..."):
                music, sr = synthesize_music(uploaded_file, vol_boost, synth_style)
                if music is not None:
                    buf = io.BytesIO()
                    sf.write(buf, music, sr, format='WAV')
                    st.audio(buf)
                    st.download_button("Download Uploaded Song", buf, "upload_song.wav")
                else:
                    st.error("No clear pitch found in this file.")

# TAB 3: TEXT TO SONG
with tab3:
    st.subheader("Convert Text to Song")
    user_text = st.text_area("Enter your lyrics/text:", "Music is the language of the soul.")
    
    if st.button("Generate Music from Text"):
        if user_text:
            with st.spinner("Converting text to speech..."):
                # Use Google Text-to-Speech
                tts = gTTS(text=user_text, lang='en')
                tts_buf = io.BytesIO()
                tts.write_to_fp(tts_buf)
                tts_buf.seek(0)
                
                st.write("Voice Preview:")
                st.audio(tts_buf)

            with st.spinner("Converting voice to music..."):
                music, sr = synthesize_music(tts_buf, vol_boost, synth_style)
                if music is not None:
                    final_buf = io.BytesIO()
                    sf.write(final_buf, music, sr, format='WAV')
                    st.subheader("Final Result")
                    st.audio(final_buf)
                    st.download_button("Download Text Song", final_buf, "text_song.wav")
