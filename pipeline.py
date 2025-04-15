import f_whisper as fw               # Real-time speech-to-text using Faster-Whisper
import text_to_text as t2t           # Text translation using CTranslate2 + SentencePiece
import text_to_speech as tts         # Text-to-speech using Piper-ONNX

# === Text-to-text translation configuration ===
T2T_SRC_LANG = "English"
T2T_TGT_LANG = "Chinese (Simplified)"
T2T_BEAM_SIZE = 3

if __name__ == "__main__":
    # Speech to text
    transcribed_input = fw.run_fast_whisper_until_silence()
    print("Transcribed:", transcribed_input)

    # Text translation
    translated_output = t2t.translate_text(
        transcribed_input, 
        T2T_SRC_LANG, 
        T2T_TGT_LANG, 
        T2T_BEAM_SIZE
    )
    print("Translated:", translated_output)

    # Text to speech
    print("Speaking translated output...")
    tts.speak(translated_output)
