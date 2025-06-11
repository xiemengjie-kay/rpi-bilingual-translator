import numpy as np
import sounddevice as sd
import sys
import time
import scipy.signal
from faster_whisper import WhisperModel
from gpiozero import Button
import signal
from collections import deque
import queue

class SpeechToTextSystem:
    def __init__(
        self,
        english_button_pin=17,
        chinese_button_pin=16,
        input_sample_rate=48000,   # mic native rate
        target_sample_rate=16000,  # whisper expects this
        channels=1,
        frames_per_buffer=1024, # Size of audio chunks to read (adjust for latency vs CPU)
        beam_size=5,
        model_size="base",
        compute_type="int8",
    ):
        # GPIO Setup
        self.english_button_pin = english_button_pin
        self.chinese_button_pin = chinese_button_pin
        self.english_button = Button(self.english_button_pin, pull_up=False)
        self.chinese_button = Button(self.chinese_button_pin, pull_up=False)

        # Audio Recording Parameters
        self.input_sample_rate = input_sample_rate
        self.target_sample_rate = target_sample_rate
        self.channels = channels
        self.frames_per_buffer = frames_per_buffer

        # Whisper Model Parameters
        self.beam_size = beam_size
        self.model_size = model_size
        self.compute_type = compute_type
        self.model = WhisperModel(self.model_size, device="cpu", compute_type=self.compute_type)

        # Recording State and Buffer
        self.is_recording = False
        self.current_language = None # To store 'en' or 'zh'
        self.audio_buffer_queue = deque()
        self.audio_stream = None

        self.transcription_results_queue = queue.Queue()

        # Assign button callbacks
        self.english_button.when_pressed = self._english_button_pressed_callback
        self.english_button.when_released = self._button_released_callback 
        self.chinese_button.when_pressed = self._chinese_button_pressed_callback
        self.chinese_button.when_released = self._button_released_callback 

        # print(f"System initialized. Monitoring GPIO pin {self.english_button_pin} for English and {self.chinese_button_pin} for Chinese.")
        # print("Press a button to start speaking, release to stop and transcribe.")
        # print("Press Ctrl+C to exit.")

    def _downsample_audio(self, audio_input):
        audio_input_f32 = audio_input.astype(np.float32)
        audio_out = scipy.signal.resample_poly(
            audio_input_f32, up=self.target_sample_rate, down=self.input_sample_rate
        )
        return audio_out

    def _transcribe_audio_buffer(self, audio_buffer_to_process, language=None, vad_parameters=None, verbose=False):
        audio_downsampled = self._downsample_audio(audio_buffer_to_process)
        segments, info = self.model.transcribe(
            audio_downsampled, beam_size=self.beam_size, language=language, vad_parameters=vad_parameters
        )
        return "".join([segment.text for segment in segments]).strip()

    def _audio_stream_callback(self, indata, frames, time_info, status):
        if status:
            print("Audio stream status:", status, file=sys.stderr)
        if self.is_recording:
            self.audio_buffer_queue.append(indata.copy())

    # Button Callbacks
    def _start_recording(self, language_code):
        if not self.is_recording:
            print(f"Button pressed! Starting recording for {language_code}...")
            self.is_recording = True
            self.current_language = language_code
            self.audio_buffer_queue.clear()

            try:
                self.audio_stream = sd.InputStream(
                    samplerate=self.input_sample_rate,
                    channels=self.channels,
                    blocksize=self.frames_per_buffer,
                    callback=self._audio_stream_callback
                )
                self.audio_stream.start()
            except Exception as e:
                print(f"Error starting audio stream: {e}")
                self.is_recording = False

    def _english_button_pressed_callback(self):
        self._start_recording('en')

    def _chinese_button_pressed_callback(self):
        self._start_recording('zh')

    def _button_released_callback(self):
        if self.is_recording:
            print(f"Button released! Stopping recording and transcribing ({self.current_language})...")
            self.is_recording = False

            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None

            if self.audio_buffer_queue:
                recorded_audio = np.concatenate(self.audio_buffer_queue, axis=0)
                recorded_audio = recorded_audio.flatten()

                print(f"Transcribing recorded audio in {self.current_language}...")
                try:
                    transcribed_text = self._transcribe_audio_buffer(
                        recorded_audio,
                        language=self.current_language
                    )
                    if transcribed_text:
                        # print(f"Transcription ({self.current_language}): ", transcribed_text)
                        self.transcription_results_queue.put(
                            {'text': transcribed_text, 'language': self.current_language}
                        )
                    else:
                        print("No speech detected or transcribed.")
                except Exception as e:
                    print(f"Error during transcription: {e}")
            else:
                print("No audio recorded.")
            self.current_language = None # Reset current language after transcription


# if __name__ == "__main__":
#     system = SpeechToTextSystem(
#         english_button_pin=17,
#         chinese_button_pin=16,
#         input_sample_rate=48000,
#         target_sample_rate=16000,
#         model_size="base",
#         compute_type="int8",
#     )
#     system.run()