#!/usr/bin/env python3
"""
Translation script using Xenova/nllb-200-distilled-600M ONNX model (merged + quantized).

Requires:
  pip install transformers optimum[onnxruntime] onnx onnxruntime matplotlib numpy
"""
import warnings
warnings.filterwarnings("ignore", message="Moving the following attributes in the config to the generation config")

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

# Mapping of human-readable names to FLoRes-200 codes
LANG_CODE_MAP = {
    "English": "eng_Latn",
    "Chinese (Simplified)": "zho_Hans",
    "Chinese (Traditional)": "zho_Hant",
    "Spanish": "spa_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Russian": "rus_Cyrl",
    "Arabic": "arb_Arab",
}

# Test sentences for each language
TEST_SENTENCES = {
    "English": "The quick brown fox jumps over the lazy dog. This is a test sentence for translation performance evaluation.",
    "Chinese (Simplified)": "敏捷的棕色狐狸跳过懒狗。这是一个用于翻译性能评估的测试句子。",
    "Spanish": "El rápido zorro marrón salta sobre el perro perezoso. Esta es una oración de prueba para evaluar el rendimiento de la traducción.",
    "Japanese": "速い茶色の狐が怠け者の犬を飛び越えます。これは翻訳性能評価のためのテスト文です。",
    "German": "Der schnelle braune Fuchs springt über den faulen Hund. Dies ist ein Testsatz zur Bewertung der Übersetzungsleistung."
}

# Global variables for model and tokenizer
_model = None
_tokenizer = None

def name_to_flores_200_code(lang_name: str) -> str:
    """Resolve FLoRes-200 language code from human-readable name."""
    lang_name = lang_name.strip().title()
    
    if "chinese" in lang_name.lower():
        if "simplified" in lang_name.lower():
            return LANG_CODE_MAP["Chinese (Simplified)"]
        elif "traditional" in lang_name.lower():
            return LANG_CODE_MAP["Chinese (Traditional)"]
        else:
            return LANG_CODE_MAP["Chinese (Simplified)"]
    
    return LANG_CODE_MAP.get(lang_name, "")

def _load_model():
    """Load the ONNX model and tokenizer if not already loaded."""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        repo = "Xenova/nllb-200-distilled-600M"
        _tokenizer = AutoTokenizer.from_pretrained(repo)
        _model = ORTModelForSeq2SeqLM.from_pretrained(
            repo,
            subfolder="onnx",
            encoder_file_name="encoder_model_quantized.onnx",
            decoder_file_name="decoder_model_merged_quantized.onnx",
            decoder_with_past_file_name="decoder_model_merged_quantized.onnx"
        )

def translate_text(
    text: str,
    src_lang: str = "English",
    tgt_lang: str = "Chinese (Simplified)",
    beam_size: int = 8,
    max_length: int = 512,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 3
) -> str:
    """
    Translate text from source language to target language.
    
    Args:
        text: Text to translate
        src_lang: Source language (default: "English")
        tgt_lang: Target language (default: "Chinese (Simplified)")
        beam_size: Number of beams for beam search (default: 8)
        max_length: Maximum length of generated text (default: 512)
        length_penalty: Length penalty for beam search (default: 1.0)
        no_repeat_ngram_size: Size of n-grams to prevent repeating (default: 3)
    
    Returns:
        Translated text
    """
    # Load model if not already loaded
    _load_model()
    
    src_code = name_to_flores_200_code(src_lang)
    tgt_code = name_to_flores_200_code(tgt_lang)
    if not src_code or not tgt_code or not text.strip():
        return ""

    _tokenizer.src_lang = src_code
    inputs = _tokenizer(text, return_tensors="pt")
    forced_bos_token_id = _tokenizer.convert_tokens_to_ids(tgt_code)

    generation_config = {
        "max_length": max_length,
        "num_beams": beam_size,
        "length_penalty": length_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "early_stopping": True,
        "forced_bos_token_id": forced_bos_token_id
    }

    outputs = _model.generate(**inputs, **generation_config)
    return _tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_stress_test(tokenizer, model, num_runs=3):
    """Run stress test across different language pairs and plot results."""
    languages = list(TEST_SENTENCES.keys())
    n_langs = len(languages)
    
    # Initialize results matrix
    times = np.zeros((n_langs, n_langs))
    std_devs = np.zeros((n_langs, n_langs))
    
    print("\nRunning stress test...")
    print(f"Testing {n_langs} languages with {num_runs} runs per pair")
    print("-" * 80)
    
    for i, src_lang in enumerate(languages):
        for j, tgt_lang in enumerate(languages):
            if src_lang == tgt_lang:
                continue
                
            run_times = []
            for _ in range(num_runs):
                start_time = time.time()
                translate_text(
                    TEST_SENTENCES[src_lang],
                    src_lang,
                    tgt_lang,
                    tokenizer,
                    model
                )
                run_times.append(time.time() - start_time)
            
            times[i, j] = np.mean(run_times)
            std_devs[i, j] = np.std(run_times)
            
            print(f"{src_lang} -> {tgt_lang}: {times[i, j]:.2f} ± {std_devs[i, j]:.2f} seconds")
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        times,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=languages,
        yticklabels=languages,
        cbar_kws={'label': 'Average Inference Time (seconds)'}
    )
    plt.title("Translation Inference Times Across Language Pairs")
    plt.xlabel("Target Language")
    plt.ylabel("Source Language")
    plt.tight_layout()
    plt.savefig("translation_performance_heatmap.png")
    print("\nPerformance heatmap saved as 'translation_performance_heatmap.png'")

if __name__ == "__main__":
    repo = "Xenova/nllb-200-distilled-600M"
    tokenizer, model = load_model(repo)
    print("Loaded merged+quantized ONNX model from subfolder 'onnx/'.")

    # Run stress test
    run_stress_test(tokenizer, model)

    # Interactive mode
    print("\nInteractive Mode:")
    print("\nAvailable languages:")
    for lang in LANG_CODE_MAP.keys():
        print(f"- {lang}")
    
    print("\nNote: For Chinese, you can use 'chinese', 'chinese simplified', or 'chinese traditional'")
    
    src = input("\nSource language (e.g., English): ")
    tgt = input("Target language (e.g., Chinese (Simplified)): ")
    txt = input("Text to translate: ")

    start_time = time.time()
    result = translate_text(
        txt,
        src,
        tgt,
        tokenizer,
        model,
        num_beams=8,
        length_penalty=1.0,
        no_repeat_ngram_size=3
    )
    end_time = time.time()
    
    print("\nTranslation:", result)
    print(f"Inference Time: {end_time - start_time:.2f} seconds")
