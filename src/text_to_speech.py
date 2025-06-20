import numpy as np
import sounddevice as sd
import tempfile
import os
import wave
from piper.voice import PiperVoice

class TextToSpeech:
    def __init__(self, voice_path="./piper_models/en_US-hfc_male-medium.onnx",
                 config_path="./piper_models/en_US-hfc_male-medium.onnx.json"):
        self.voice_path = voice_path
        self.config_path = config_path
        self.voice = self.init_tts(self.voice_path, self.config_path)

    def init_tts(self, voice_path, config_path):
        try:
            return PiperVoice.load(voice_path, config_path)
        except Exception as e:
            print(f"[TTS INIT ERROR] {e}")
            return None

    def set_voice(self, voice_path, config_path):
        self.voice_path = voice_path
        self.config_path = config_path
        self.voice = self.init_tts(self.voice_path, self.config_path)

    def synthesize_audio(self, text):
        if not text.strip() or self.voice is None:
            return None, None
        try:
            temp_wav_fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(temp_wav_fd)
            with wave.open(temp_wav_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(22050)
                wav_file.tell()
                self.voice.synthesize(text.strip(), wav_file=wav_file)
            with wave.open(temp_wav_path, "rb") as wf:
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                audio_frames = wf.readframes(n_frames)
            os.remove(temp_wav_path)
            audio = np.frombuffer(audio_frames, dtype=np.int16).astype(np.float32)
            audio = audio * (1.0 / 32768.0)
            return audio, sample_rate
        except Exception as e:
            print(f"[TTS SYNTH ERROR] {e}")
            if 'temp_wav_path' in locals() and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
            return None, None

    def play_audio(self, audio, sample_rate):
        if audio is not None and sample_rate is not None:
            try:
                sd.play(audio, samplerate=sample_rate, blocking=True)
            except Exception as e:
                print(f"[AUDIO ERROR] {e}")

    def speak(self, text):
        audio, sample_rate = self.synthesize_audio(text)
        if audio is not None and sample_rate is not None:
            self.play_audio(audio, sample_rate)
            print("Audio playback complete.")

if __name__ == "__main__":
    tts = TextToSpeech()
    tts.speak("hello, how are you doing today?")
