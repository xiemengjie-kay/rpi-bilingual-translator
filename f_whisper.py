import numpy as np
import sounddevice as sd
import sys
import time
from faster_whisper import WhisperModel

# === Config ===
SAMPLE_RATE = 16000
CHANNELS = 1
BUFFER_DURATION = 5  # seconds
BEAM_SIZE = 5
MODEL_SIZE = "tiny"
COMPUTE_TYPE = "int8"

# === Global audio buffer ===
BUFFER_SAMPLES = BUFFER_DURATION * SAMPLE_RATE
audio_buffer = np.zeros((BUFFER_SAMPLES,), dtype=np.float32)

# === Callback to update rolling audio buffer ===
def update_audio_buffer(indata, outdata, frames, time_info, status):
    global audio_buffer
    if status:
        print("Sound device error:", status, file=sys.stderr)
    indata_f32 = indata.flatten()
    audio_buffer = np.concatenate((audio_buffer, indata_f32)).flatten()
    audio_buffer = audio_buffer[len(indata_f32):]

# === Run the transcription model ===
def run_model(model, audio_input, beam_size, vad_parameters=None, verbose=True):
    transcribe_start_time = time.time()
    segments, info = model.transcribe(audio_input, beam_size=beam_size, vad_parameters=vad_parameters)
    transcribe_end_time = time.time()
    transcribe_duration = transcribe_end_time - transcribe_start_time

    if verbose:
        print("Transcribe time: {:.0f}ms".format(transcribe_duration * 1000))
        print("Detected language '%s' with probability %f" %
                (info.language, info.language_probability))

    all_text = ""
    segment_start_time = time.time()
    for segment in segments:
        if verbose:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        else:
            all_text += segment.text
    segment_end_time = time.time()
    segment_duration = segment_end_time - segment_start_time
    if verbose:
        print("Segment time: {:.0f}ms".format(segment_duration * 1000))
    else:
        print(all_text)

# === Main runner ===
def run_fast_whisper():
    global audio_buffer

    print("Loading model...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)
    print("Model ready. Listening... (Press Ctrl+C to stop)")

    sd.default.samplerate = SAMPLE_RATE
    sd.default.channels = CHANNELS

    with sd.Stream(callback=update_audio_buffer):
        while True:
            time.sleep(BUFFER_DURATION)
            run_model(
                model,
                audio_buffer,
                BEAM_SIZE,
                vad_parameters={
                    "threshold": 0.0,
                    "min_silence_duration_ms": BUFFER_DURATION * 1000,
                    "min_speech_duration_ms": BUFFER_DURATION * 1000,
                },
                verbose=False
            )

# === Entry point ===
if __name__ == "__main__":
    run_fast_whisper()
