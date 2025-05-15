import numpy as np
import sounddevice as sd
import tempfile
import os
import wave
from piper.voice import PiperVoice

class TextToSpeech:
    def __init__(self, voice_path="./piper_models/zh_CN-huayan-medium.onnx",
                 config_path="./piper_models/zh_CN-huayan-medium.onnx.json"):
        """
        Initialize the TTS engine with the given model and config paths.
        """
        self.voice_path = voice_path
        self.config_path = config_path
        self.voice = self.init_tts(self.voice_path, self.config_path)

    def init_tts(self, voice_path, config_path):
        """
        Load and return a PiperVoice instance from model and config files.
        """
        try:
            return PiperVoice.load(voice_path, config_path)
        except Exception as e:
            print(f"[TTS INIT ERROR] {e}")
            return None

    def set_voice(self, voice_path, config_path):
        """
        Change the voice model and config.
        """
        self.voice_path = voice_path
        self.config_path = config_path
        self.voice = self.init_tts(self.voice_path, self.config_path)

    def synthesize_audio(self, text):
        """
        Convert input text to float32 audio by synthesizing to a temporary WAV file,
        then reading and normalizing the output.
        """
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
        """
        Play the normalized audio buffer using sounddevice.
        """
        if audio is not None and sample_rate is not None:
            try:
                sd.play(audio, samplerate=sample_rate, blocking=True)
            except Exception as e:
                print(f"[AUDIO ERROR] {e}")

    def speak(self, text):
        """
        Run the full text-to-speech pipeline: synthesize and play audio.
        """
        audio, sample_rate = self.synthesize_audio(text)
        if audio is not None and sample_rate is not None:
            self.play_audio(audio, sample_rate)
            print("Audio playback complete.")

if __name__ == "__main__":
    tts = TextToSpeech()
    tts.speak("你好，你今天怎么样")
    # To change voice:
    # tts.set_voice("path/to/other/model.onnx", "path/to/other/model.onnx.json")
    # tts.speak("用新声音说话")
