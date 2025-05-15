import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
import torchaudio
import torch
import subprocess
import time
import soundfile as sf
import time
import tempfile
from transformers import AutoProcessor, SeamlessM4TModel

DEVICE = "cpu"
MODEL_NAME = "facebook/hf-seamless-m4t-medium"
#seamless-m4t-small
TARGET_LANG = "cmn"  # Mandarin Chinese
AUDIO_INPUT_FILE = "mic_input.wav"
AUDIO_OUTPUT_FILE = "mic_output.wav"
AUDIO_INPUT = "plughw:0,0"
AUDIO_OUTPUT = 0

def record_with_arecord(filename="mic_input.wav", duration=5, device="plughw:0,0"):
    """
    Record audio from microphone using arecord.
    """
    print(f"🎙 Recording for {duration} seconds...")
    subprocess.run([
        "arecord", "-D", device, "-f", "S16_LE", "-r", "16000", "-c", "1",
        "-d", str(duration), filename
    ], check=True)
    print("Recording complete.")

def load_waveform(filename):
    waveform, sample_rate = torchaudio.load(filename)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    waveform = waveform.mean(0) if waveform.shape[0] > 1 else waveform[0]
    return waveform

def run_translation(waveform):
    inputs = processor(audios=waveform, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    audio_array = model.generate(**inputs, tgt_lang=TARGET_LANG)[0].cpu().numpy().squeeze()
    return audio_array
    
# def load_audio(filename):
#     """
#     Load audio from WAV file and convert to normalized float32 array.
#     """
#     rate, data = wavfile.read(filename)
#     assert rate == 16000, "Expected 16kHz sample rate"
#     audio = data.astype(np.float32) / 32768.0  # Normalize int16 to float32
#     return audio

def play_audio(audio_array, sample_rate=16000, filename=AUDIO_OUTPUT_FILE):
    # with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
    sf.write(filename, audio_array, sample_rate, subtype='PCM_16')
    subprocess.run(["aplay", "-D", "plughw:0,0", filename], check=True)

# Load processor and model
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = SeamlessM4TModel.from_pretrained(MODEL_NAME)
# model = ORTModelForSpeechSeq2Seq.from_pretrained("facebook/hf-seamless-m4t-medium")

device = torch.device(DEVICE)
model.to(device)

def main():
    # print(sd.query_devices())
    record_with_arecord(device=AUDIO_INPUT)
    waveform = load_waveform(AUDIO_INPUT_FILE)
    start_time = time.perf_counter()
    audio_array_from_audio = run_translation(waveform)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Preprocess and Inference execution time: {execution_time:.4f} seconds")
    play_audio(audio_array_from_audio)


if __name__ == "__main__":
    while(True):
        main()
        time.sleep(1); # sleep for 1s