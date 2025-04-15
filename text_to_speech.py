import numpy as np
import sounddevice as sd
import tempfile
import os
import wave
from piper.voice import PiperVoice

def init_tts(voice_path="./piper_models/zh_CN-huayan-medium.onnx",
             config_path="./piper_models/zh_CN-huayan-medium.onnx.json"):
    """
    Load and return a PiperVoice instance from model and config files.
    """
    try:
        return PiperVoice.load(voice_path, config_path)
    except Exception as e:
        print(f"[TTS INIT ERROR] {e}")
        return None

def synthesize_audio(voice, text):
    """
    Convert input text to float32 audio by synthesizing to a temporary WAV file,
    then reading and normalizing the output.
    """
    if not text.strip():
        return None, None
    try:
        # Create a temporary file and get its path
        temp_wav_fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_wav_fd)  # Close the low-level file descriptor
        
        # Create a wave file object with the appropriate settings
        with wave.open(temp_wav_path, "wb") as wav_file:
            # Initialize with some default settings that will be overwritten by voice.synthesize
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(22050)  # Default rate, will likely be changed
            
            # Rewind to the beginning of the file after initialization
            wav_file.tell()
            
            # Synthesize audio into the wave file
            voice.synthesize(text.strip(), wav_file=wav_file)
        
        # Read the generated WAV file
        with wave.open(temp_wav_path, "rb") as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            audio_frames = wf.readframes(n_frames)
        
        # Remove the temporary file
        os.remove(temp_wav_path)
        
        # Convert raw audio data to a numpy array and normalize it
        audio = np.frombuffer(audio_frames, dtype=np.int16).astype(np.float32)
        audio = audio * (1.0 / 32768.0)
        
        return audio, sample_rate
    except Exception as e:
        print(f"[TTS SYNTH ERROR] {e}")
        # Clean up the temp file if it exists
        if 'temp_wav_path' in locals() and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        return None, None

def play_audio(audio, sample_rate):
    """
    Play the normalized audio buffer using sounddevice.
    """
    if audio is not None and sample_rate is not None:
        try:
            sd.play(audio, samplerate=sample_rate, blocking=True)
        except Exception as e:
            print(f"[AUDIO ERROR] {e}")

def speak(text, voice=None):
    """
    Run the full text-to-speech pipeline: synthesize and play audio.
    """
    voice = voice or init_tts()
    if voice is None:
        return
    audio, sample_rate = synthesize_audio(voice, text)
    if audio is not None and sample_rate is not None:
        play_audio(audio, sample_rate)
        print("Audio playback complete.")
