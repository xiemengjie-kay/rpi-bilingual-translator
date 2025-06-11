#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sacrebleu
from datasets import load_dataset
from tqdm import tqdm

from text_to_text_small100 import Translator_text


# In[3]:


dataset = load_dataset("opus100", "en-zh")  # 1M+ sentence pairs


# In[ ]:


translate_text = Translator_text()
# start_time = time.time()
# print(translate_text(english_text, "en", "zh"))
# end_time = time.time()
# print(f"""Inference time: {end_time-start_time:.2f}s""")


# In[ ]:


def prepare_batch(batch):
    src_texts = batch['translation']['en']
    tgt_texts = batch['translation']['zh']
    preds = translate_text(src_texts, source_language="en", target_language="zh")
    return preds, tgt_texts


# In[ ]:


predictions = []
references = []


# In[ ]:


for sample in tqdm(dataset["test"], desc="Translating"):
    pred, ref = prepare_batch(sample)
    # print(pred)
    # print(type(pred))
    # print(ref)
    # print(type(ref))
    predictions.append(pred)
    references.append(ref)  # Each reference must be a list of one or more refs


# In[ ]:


bleu = sacrebleu.corpus_bleu(predictions, [references])
print(f"\n📈 BLEU Score: {bleu.score:.5f}")


# In[ ]:


for i in range(5):
    print(f"\nExample {i+1}:")
    print(f"English Input: {dataset['test'][i]['translation']['en']}")
    print(f"Reference Chinese: {references[i]}")
    print(f"Model Prediction: {predictions[i]}")

