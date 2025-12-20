import streamlit as st
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

st.set_page_config(page_title="SoundSense", layout="wide")
st.title("🎧 SoundSense – Inclusive AI Audio App")

choice = st.sidebar.selectbox(
    "Select Module",
    ["Voice → Music", "Music Visualizer", "Sound Alerts"]
)

# ---------- MODULE 1: VOICE TO MUSIC ----------
if choice == "Voice → Music":
    st.header("🎤 Voice to Music Converter")
    st.write("Upload a recording of someone speaking or singing. The AI will extract the pitch and convert it into a musical instrument melody.")

    audio_file = st.file_uploader("Upload human voice (.wav/.mp3)", type=["wav","mp3"])

    if audio_file:
        # Load the audio
        y, sr = librosa.load(audio_file)
        
        with st.spinner('Converting your voice to music...'):
            # 1. Pitch Extraction (Finding the melody of the voice)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7')
            )
            
            # Replace NaN (silence) with zeros
            f0 = np.nan_to_num(f0)

            if np.all(f0 == 0):
                st.warning("⚠️ No clear human voice/pitch detected! Try a clearer recording.")
            else:
                # 2. Quantization (The "Auto-tune" Step)
                # Map raw frequencies to the nearest musical MIDI notes
                voiced_indices = f0 > 0
                midi_notes = librosa.hz_to_midi(f0[voiced_indices])
                quantized_midi = np.round(midi_notes) # Rounding to nearest semi-tone
                f0[voiced_indices] = librosa.midi_to_hz(quantized_midi)
                
                # 3. Synthesis (Creating the Instrument Sound)
                # We create a richer sound by combining a base wave with its harmonics
                times = np.arange(len(y)) / sr
                music_signal = np.zeros_like(y)
                
                # Resample f0 to match the length of the original audio signal
                # (pyin gives f0 at a lower hop_length)
                f0_resampled = np.interp(
                    np.linspace(0, len(f0), len(y)), 
                    np.arange(len(f0)), 
                    f0
                )

                # Generate sound based on the extracted pitch
                phase = np.cumsum(2 * np.pi * f0_resampled / sr)
                # Mix Sine wave + some harmonics for a "flute/synth" feel
                music_signal = 0.5 * np.sin(phase) + 0.2 * np.sin(2 * phase) 
                
                # Cleanup: remove sound where there was no voice
                music_signal[f0_resampled == 0] = 0
                
                # Normalize volume
                music_signal = music_signal / (np.max(np.abs(music_signal)) + 1e-6)
                
                # 4. Display Results
                sf.write("voice_to_music.wav", music_signal, sr)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Voice")
                    st.audio(audio_file)
                with col2:
                    st.subheader("AI Generated Music")
                    st.audio("voice_to_music.wav")
                
                st.success("✅ Conversion Complete!")

# ---------- MODULE 2: VISUALIZER ----------
elif choice == "Music Visualizer":
    st.header("🎶 Music Visualizer")
    music = st.file_uploader("Upload music", type=["wav","mp3"])

    if music:
        y, sr = librosa.load(music)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        st.write(f"Detected Tempo: **{round(float(tempo), 1)} BPM**")

        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr, ax=ax, color='purple')
        st.pyplot(fig)

# ---------- MODULE 3: ALERTS ----------
elif choice == "Sound Alerts":
    st.header("🚨 Sound Alerts")
    sound = st.file_uploader("Upload sound", type=["wav","mp3"])

    if sound:
        y, sr = librosa.load(sound)
        rms = np.mean(librosa.feature.rms(y=y))
        if rms > 0.05:
            st.error("⚠️ Loud Sound Detected!")
        else:
            st.success("Environment is Quiet")
