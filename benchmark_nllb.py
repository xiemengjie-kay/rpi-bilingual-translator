#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Benchmark Fine-Tuned NLLB-200 Model for English-Chinese Translation using BLEU


# In[2]:


import sacrebleu
import ctranslate2
from datasets import load_dataset
from tqdm import tqdm

import sys
import os
sys.path.append(os.getcwd())  # Adds current notebook's directory

from text_to_text import translate_text


# In[3]:


dataset = load_dataset("opus100", "en-zh")  # 1M+ sentence pairs


# In[4]:


def prepare_batch(batch):
    src_texts = batch['translation']['en']
    tgt_texts = batch['translation']['zh']
    preds = translate_text(src_texts, "English", "Chinese (Simplified)", beam_size=5)
    return preds, tgt_texts


# In[15]:


predictions = []
references = []


# In[ ]:


for sample in tqdm(dataset["test"].select(range(2)), desc="Translating"):
    pred, ref = prepare_batch(sample)
    # print(pred)
    # print(type(pred))
    # print(ref)
    # print(type(ref))
    predictions.append(pred)
    references.append(ref)  # Each reference must be a list of one or more refs


# In[ ]:


# print(predictions)


# In[ ]:


# print(references)


# In[19]:


bleu = sacrebleu.corpus_bleu(predictions, [references])
print(f"\n📈 BLEU Score: {bleu.score:.5f}")


# In[ ]:


for i in range(2):
    print(f"\nExample {i+1}:")
    print(f"English Input: {dataset['test'][i]['translation']['en']}")
    print(f"Reference Chinese: {references[i]}")
    print(f"Model Prediction: {predictions[i]}")

