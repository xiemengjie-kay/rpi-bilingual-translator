import numpy as np
import sounddevice as sd
import sys
import time
from faster_whisper import WhisperModel  # Optimized ASR model for fast, accurate transcription

# === Audio callback: updates a fixed-size rolling buffer with live microphone input ===
def update_audio_buffer(indata, outdata, frames, time_info, status):
    global audio_buffer
    if status:
        print("Sound device error:", status, file=sys.stderr)

    # Flatten input and roll into buffer (FIFO-style)
    indata_f32 = indata.flatten()
    audio_buffer = np.concatenate((audio_buffer, indata_f32)).flatten()
    audio_buffer = audio_buffer[len(indata_f32):]

# === Run Whisper model on current audio buffer and return combined transcript ===
def run_model(model, audio_input, beam_size, vad_parameters=None, verbose=False):
    segments, info = model.transcribe(
        audio_input, beam_size=beam_size, vad_parameters=vad_parameters
    )
    return "".join([segment.text for segment in segments]).strip()

# === Main loop: listen for speech and stop after a period of silence ===
def run_fast_whisper_until_silence(
    sample_rate=16000,          # Sampling rate in Hz
    channels=1,                 # Mono audio
    buffer_duration=5,          # Rolling buffer size in seconds
    beam_size=5,                # Decoding beam width (higher = more accurate, slower)
    model_size="tiny",          # Whisper model variant (e.g., "tiny", "base")
    compute_type="int8",        # Quantized inference (low-resource optimization)
    silence_timeout=1.2         # Exit after N seconds of silence
):
    global audio_buffer

    buffer_samples = buffer_duration * sample_rate
    audio_buffer = np.zeros((buffer_samples,), dtype=np.float32)

    # Initialize Whisper ASR model
    model = WhisperModel(model_size, device="cpu", compute_type=compute_type)

    # Configure audio stream
    sd.default.samplerate = sample_rate
    sd.default.channels = channels

    print("Listening... Speak now.")

    full_text = ""
    last_nonempty_result_time = time.time()

    # Stream from microphone and process buffer repeatedly
    with sd.Stream(callback=update_audio_buffer):
        try:
            while True:
                time.sleep(buffer_duration)  # Wait for buffer to fill

                # Transcribe current audio buffer
                result = run_model(
                    model,
                    audio_buffer,
                    beam_size,
                    vad_parameters={
                        "threshold": 0.5,                          # VAD sensitivity
                        "min_silence_duration_ms": silence_timeout * 1000,
                        "min_speech_duration_ms": 250,            # Filter short bursts
                    },
                    verbose=False
                )

                if result:
                    print("Recognized:", result)
                    full_text += result + " "
                    last_nonempty_result_time = time.time()
                elif time.time() - last_nonempty_result_time > silence_timeout:
                    print("Stopped (silence detected).")
                    break
        except KeyboardInterrupt:
            print("\nManually stopped by user.")

    return full_text.strip()

# === Script entry point ===
if __name__ == "__main__":
    final_text = run_fast_whisper_until_silence()
    print("\nFinal recognized text:", final_text)