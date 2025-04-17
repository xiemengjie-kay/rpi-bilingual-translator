import ctranslate2
import sentencepiece as spm

# === Mapping of human-readable language names to FLoRes-200 codes ===
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

def name_to_flores_200_code(lang_name):
    """Resolve FLoRes-200 language code from human-readable name."""
    return LANG_CODE_MAP.get(lang_name)

# === Model and tokenizer paths ===
MODEL_DIR = "./models/nllb-200-distilled-600M-int8/"
SP_MODEL_PATH = f"{MODEL_DIR}/flores200_sacrebleu_tokenizer_spm.model"

# === Load sentencepiece tokenizer and CTranslate2 translator once ===
sp = spm.SentencePieceProcessor(model_file=SP_MODEL_PATH)
translator = ctranslate2.Translator(MODEL_DIR, device="cpu", compute_type="int8")

def translate_text(input_text: str, source_language: str, target_language: str, beam_size: int = 1) -> str:
    """
    Translates input_text from source_language to target_language using CTranslate2.
    
    Args:
        input_text (str): Text to translate.
        source_language (str): Source language (e.g., "English").
        target_language (str): Target language (e.g., "Chinese (Simplified)").
        beam_size (int): Beam size for decoding (default=1).
    
    Returns:
        str: Translated output text.
    """
    src_lang = name_to_flores_200_code(source_language)
    tgt_lang = name_to_flores_200_code(target_language)
    if not src_lang or not tgt_lang or not input_text.strip():
        return ""

    # Tokenize input and prepend source language code
    tokens = sp.encode_as_pieces(input_text.strip())
    tokens = [[src_lang] + tokens + ["</s>"]]

    # Prefix target language token for guided decoding
    target_prefix = [[tgt_lang]]

    # Perform batched translation (single example in this case)
    result = translator.translate_batch(
        tokens,
        beam_size=beam_size,
        max_batch_size=1,
        batch_type="tokens",
        target_prefix=target_prefix
    )

    # Extract output tokens and remove target language token if present
    output_tokens = result[0].hypotheses[0]
    if tgt_lang in output_tokens:
        output_tokens.remove(tgt_lang)

    return sp.decode(output_tokens)

# === Sample usage for quick testing ===
if __name__ == "__main__":
    output = translate_text("Hey how are you?", "English", "Chinese (Simplified)", beam_size=5)
    print("Translated:", output)
