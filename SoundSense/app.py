import streamlit as st
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import io

st.set_page_config(page_title="SoundSense", layout="wide")
st.title("🎧 SoundSense – Inclusive AI Audio App")

choice = st.sidebar.selectbox(
    "Select Module",
    ["Voice → Music", "Music Visualizer", "Sound Alerts"]
)

# ---------- MODULE 1: VOICE TO MUSIC ----------
if choice == "Voice → Music":
    st.header("🎤 Voice to Music Converter")
    
    audio_file = st.file_uploader("Upload human voice (.wav/.mp3)", type=["wav","mp3"])

    if audio_file:
        # 1. Load the audio carefully
        # We use sr=None to keep the original quality
        y, sr = librosa.load(audio_file, sr=22050) 
        
        st.info("Processing your voice...")

        # 2. Extract Pitch (f0)
        # We adjust parameters to be more sensitive to human speech
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            fill_na=0 # Fill silent parts with 0
        )

        # 3. Check if any pitch was detected
        if np.sum(f0) == 0:
            st.error("❌ Could not detect a clear voice. Please speak louder or closer to the mic.")
        else:
            # 4. Synthesize Music Signal
            # We must make sure the output length matches the input
            # pyin uses a default hop_length of 512. We reconstruct based on that.
            hop_length = 512
            # Create a continuous frequency array to match the original audio length
            f0_stretched = np.repeat(f0, hop_length)
            
            # Ensure the lengths match exactly
            if len(f0_stretched) > len(y):
                f0_stretched = f0_stretched[:len(y)]
            else:
                f0_stretched = np.pad(f0_stretched, (0, len(y) - len(f0_stretched)))

            # Generate the phase (the "movement" of the sound)
            # Dividing by sr is crucial for correct timing
            phase = np.cumsum(2 * np.pi * f0_stretched / sr)
            
            # Create a Sine wave + a Harmonic (to make it sound like an instrument)
            music_signal = 0.5 * np.sin(phase) + 0.25 * np.sin(2 * phase)
            
            # Remove sound from silent/unvoiced parts
            music_signal[f0_stretched == 0] = 0
            
            # Normalize volume (Prevent clipping/distortion)
            if np.max(np.abs(music_signal)) > 0:
                music_signal = music_signal / np.max(np.abs(music_signal))

            # 5. Export to a Buffer (Memory) instead of a file for better stability
            buffer = io.BytesIO()
            sf.write(buffer, music_signal, sr, format='WAV')
            buffer.seek(0)

            # 6. Display Results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Voice")
                st.audio(audio_file)
            with col2:
                st.subheader("AI Generated Music")
                st.audio(buffer, format='audio/wav')
            
            st.success("✅ Music generated successfully!")

# ... (Keep the rest of your Visualizer and Alert modules as they were)
