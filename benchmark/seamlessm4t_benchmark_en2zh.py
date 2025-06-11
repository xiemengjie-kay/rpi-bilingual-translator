#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datasets import load_dataset
from transformers import AutoProcessor, SeamlessM4TModel
import torch, time, numpy as np
import evaluate
from tqdm import tqdm


# In[3]:


device = 'cpu'
# Load model and processor
model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium").to(device)
processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")


# In[8]:


# Load a small split of the CVSS English-Chinese test set
dataset = load_dataset("facebook/covost2", "en_zh-CN", split="test[:10]")  # Loading first 10 samples for testing


# In[ ]:


# Optional: Load BLEU scorer for intermediate text accuracy
bleu = evaluate.load("bleu")


# In[ ]:


results = []

for sample in tqdm(dataset):
    audio = sample['audio']['array']
    sr = sample['audio']['sampling_rate']
    reference_text = sample['translation']['zh']

    inputs = processor(audios=audio, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    start = time.time()
    outputs = model.generate(**inputs, tgt_lang="cmn")
    end = time.time()

    predicted_audio = outputs[0].cpu().numpy().squeeze()
    input_dur = len(audio) / sr
    output_dur = len(predicted_audio) / 16000
    latency = end - start

    results.append({
        "input_duration": input_dur,
        "output_duration": output_dur,
        "latency": latency,
        "RTF": latency / input_dur,
    })


# In[ ]:


# Display results
import pandas as pd
df = pd.DataFrame(results)
df.describe()

