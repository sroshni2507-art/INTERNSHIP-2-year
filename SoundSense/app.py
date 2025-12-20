import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import io

# App Configuration
st.set_page_config(page_title="AI Voice Songwriter", layout="wide", page_icon="🎤")
st.title("🎤 AI Live Voice-to-Song")
st.write("Speak or hum into the microphone, and this AI will turn your voice into a musical melody with a beat.")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Musical Controls")
vol_boost = st.sidebar.slider("Volume Boost", 1.0, 10.0, 4.0)
instrument = st.sidebar.selectbox("Instrument Style", ["Synthesizer", "Electronic Lead", "Clean Sine"])
add_drums = st.sidebar.checkbox("Add Background Beat", value=True)

# Helper Function: Snaps voice pitch to the nearest musical note (Auto-tune)
def auto_tune(f0):
    if f0 <= 0: return 0
    # Frequencies for C Major Scale (Hz)
    c_major_hz = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
    return min(c_major_hz, key=lambda x: abs(x - f0))

# Core Engine: Converts Voice to Song
def convert_voice_to_song(audio_data, boost, style):
    # Load the recorded audio
    y, sr = librosa.load(audio_data, sr=None)
    
    # 1. Extract Pitch (f0) and Volume (RMS)
    hop_length = 512
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=80, fmax=800, sr=sr)
    
    # Volume envelope (So the music stops when you stop talking)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms = np.interp(np.arange(len(f0)), np.arange(len(rms)), rms)
    rms = rms / (np.max(rms) + 1e-6) # Normalize

    # 2. Apply Auto-tune
    f0_tuned = np.array([auto_tune(f) for f in f0])
    
    # 3. Match length (Upsampling)
    total_samples = len(f0) * hop_length
    f0_final = np.interp(np.arange(total_samples), np.arange(0, total_samples, hop_length), f0_tuned)
    rms_final = np.interp(np.arange(total_samples), np.arange(0, total_samples, hop_length), rms)

    # 4. Synthesize Waveform
    phase = 2 * np.pi * np.cumsum(f0_final) / sr
    
    if style == "Synthesizer":
        # Mix Sine and Square for a rich synth sound
        music = 0.6 * np.sin(phase) + 0.3 * np.sign(np.sin(phase))
    elif style == "Electronic Lead":
        music = np.sign(np.sin(phase)) # Pure Square wave
    else:
        music = np.sin(phase) # Pure Sine wave

    # Apply Volume Envelope and Gain
    music = music * rms_final * boost
    music = np.clip(music, -1.0, 1.0) 

    # 5. Add a simple Background Beat (Drum kick)
    if add_drums:
        tempo_hz = 2.0  # 120 BPM
        t = np.arange(total_samples) / sr
        kick = np.sin(2 * np.pi * 60 * t) * np.exp(-12 * (t % (1/tempo_hz)))
        music = (0.7 * music) + (0.3 * kick)

    return music, sr

# --- USER INTERFACE ---
st.subheader("Step 1: Record your voice")
user_voice = st.audio_input("Click to Record")

if user_voice:
    st.write("Voice Recording Found!")
    if st.button("✨ Convert My Voice into a Song"):
        with st.spinner("Processing pitch and applying Auto-tune..."):
            song, sr_out = convert_voice_to_song(user_voice, vol_boost, instrument)
            
            if song is not None:
                # Save to memory buffer
                out_buffer = io.BytesIO()
                sf.write(out_buffer, song, sr_out, format='WAV')
                out_buffer.seek(0)
                
                st.divider()
                st.subheader("Step 2: Listen to your AI Song")
                st.audio(out_buffer)
                st.download_button(
                    label="Download Your Song",
                    data=out_buffer,
                    file_name="my_ai_voice_song.wav",
                    mime="audio/wav"
                )
            else:
                st.error("The recording was too quiet. Please try again.")
