import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import io
import speech_recognition as sr
from gtts import gTTS

st.set_page_config(page_title="AI Music Composer", layout="wide", page_icon="🎵")
st.title("🎵 AI Voice-to-Song Composer")
st.write("Convert your voice into a musical melody with Auto-Tune and Dynamics.")

# -------- SIDEBAR --------
st.sidebar.header("Musical Settings")
tempo = st.sidebar.slider("Song Tempo (BPM)", 60, 180, 120)
instrument = st.sidebar.selectbox("Instrument Sound", ["Piano/Synth", "Electric Lead", "Soft Pad"])

# -------- HELPER FUNCTIONS --------

def quantize_to_scale(f0):
    """Snaps frequencies to the nearest musical notes in C Major scale."""
    if f0 <= 0: return 0
    # Standard frequencies for C Major Scale
    # C4, D4, E4, F4, G4, A4, B4
    scale_hz = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]
    return min(scale_hz, key=lambda x: abs(x - f0))

def synthesize_musical_song(audio_input, instrument_type):
    y, sr = librosa.load(audio_input, sr=None)
    
    # 1. Pitch Extraction
    hop_length = 512
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=80, fmax=800, sr=sr)
    
    # 2. Extract Volume Envelope (So it's not a constant BEEEP)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms = np.interp(np.arange(len(f0)), np.arange(len(rms)), rms)
    rms = rms / (np.max(rms) + 1e-6) # Normalize volume

    # 3. Clean and Auto-Tune the Pitch
    f0_musical = np.array([quantize_to_scale(f) for f in f0])
    
    # Upsample to full audio length
    total_samples = len(f0) * hop_length
    f0_final = np.interp(np.arange(total_samples), np.arange(0, total_samples, hop_length), f0_musical)
    rms_final = np.interp(np.arange(total_samples), np.arange(0, total_samples, hop_length), rms)

    # 4. Generate the Sound
    phase = 2 * np.pi * np.cumsum(f0_final) / sr
    
    if instrument_type == "Piano/Synth":
        # Mix of Sine and Sawtooth for a richer sound
        music = 0.6 * np.sin(phase) + 0.3 * (2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5)))
    elif instrument_type == "Electric Lead":
        music = np.sign(np.sin(phase)) * 0.5 # Square wave
    else:
        music = np.sin(phase) # Pure Sine

    # 5. Apply the Volume Envelope (Silence when you aren't talking)
    music = music * rms_final 
    
    # Soft Fade in/out to remove clicks
    music = np.clip(music * 2.0, -1.0, 1.0) 
    
    return music, sr

# -------- APP FLOW --------

tab1, tab2 = st.tabs(["🎤 Record My Voice", "📝 Text to Song"])

with tab1:
    st.subheader("1. Record your Voice/Song")
    rec = st.audio_input("Speak or Sing")
    
    if rec:
        if st.button("Convert to Musical Song"):
            with st.spinner("Applying Auto-Tune and Harmonizing..."):
                music, sample_rate = synthesize_musical_song(rec, instrument)
                if music is not None:
                    buf = io.BytesIO()
                    sf.write(buf, music, sample_rate, format='WAV')
                    st.success("Your song is ready!")
                    st.audio(buf)
                    st.download_button("Download Song", buf, "my_ai_song.wav")

with tab2:
    st.subheader("1. Type Lyrics")
    txt = st.text_area("Enter your song lyrics:", "I love making music with AI.")
    
    if st.button("Convert Lyrics to Song"):
        with st.spinner("Generating AI Voice..."):
            tts = gTTS(text=txt, lang='en')
            t_buf = io.BytesIO()
            tts.write_to_fp(t_buf)
            t_buf.seek(0)
            
            music, sample_rate = synthesize_musical_song(t_buf, instrument)
            if music is not None:
                final_buf = io.BytesIO()
                sf.write(final_buf, music, sample_rate, format='WAV')
                st.audio(final_buf)
