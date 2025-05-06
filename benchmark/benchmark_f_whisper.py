#!/usr/bin/env python
# coding: utf-8

# In[2]:


### Benchmark f_whisper Model for English-Chinese Speech-to-Text Translation using BLEU


# In[24]:


import torch
import torchaudio
import librosa
import sacrebleu
import numpy as np
from faster_whisper import WhisperModel
from datasets import load_dataset
from tqdm import tqdm


# In[4]:


silence_timeout = 1.2


# In[5]:


dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="test")


# In[6]:


model_size="tiny"
model = WhisperModel(model_size, device="cpu", compute_type="int8")


# In[27]:


predictions = []
references = []


# In[8]:


def prepare_audio(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        speech_array = resampler(speech_array)
    batch["speech"] = speech_array.squeeze(0)
    return batch


# In[9]:


if "path" in dataset.column_names:
    dataset = dataset.map(prepare_audio)


# In[28]:


# for i in tqdm(range(len(dataset)), desc="Transcribing"):
for i in tqdm(range(2000), desc="Transcribing"):
    sample = dataset[i]
    # print(sample.keys())
    # print(np.asarray(sample["speech"]))
    if "speech" in sample:
        waveform = np.asarray(sample["speech"])
        segments, info = model.transcribe(waveform, 
                                          language="en", 
                                          beam_size=3, 
                                          vad_parameters={"threshold": 0.5,                          # VAD sensitivity
                                                          "min_silence_duration_ms": silence_timeout * 1000,
                                                          "min_speech_duration_ms": 250,            # Filter short bursts
                                                          }
        )
        transcription = "".join([segment.text for segment in segments]).strip()
    
        predictions.append(transcription)
        references.append(sample.get("sentence", ""))
        # print(predictions)
        # print(references)


# In[14]:


# print(predictions)


# In[12]:


bleu = sacrebleu.corpus_bleu(predictions, [references])
print(f"\nBLEU Score: {bleu.score:.2f}")


# In[ ]:


for i in range(5):
    print(f"\nExample {i+1}:")
    print(f"Reference English: {references[i]}")
    print(f"Model Prediction: {predictions[i]}")

