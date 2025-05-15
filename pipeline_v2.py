import f_whisper                             # Real-time speech-to-text using Faster-Whisper
import text_to_text_opus_en_zh as t2t_en_zh  # Text English-to-Chinese translation
import text_to_speech as tts                 # Text-to-speech using Piper-ONNX

if __name__ == "__main__":
    # Speech to text
    fw = f_whisper.WhisperRecorder()
    translator_en_zh = t2t_en_zh.EnglishToChineseTranslator()
    zh_speaker = tts.TextToSpeech(voice_path="./piper_models/zh_CN-huayan-medium.onnx",
                                  config_path="./piper_models/zh_CN-huayan-medium.onnx.json")
    en_speaker = tts.TextToSpeech(voice_path="./piper_models/en_US-amy-medium.onnx",
                                  config_path="./piper_models/en_US-amy-medium.onnx.json")
    
    
    transcribed_input = fw.run()
    print("Transcribed:", transcribed_input)
    
    # English to Chinese translation
    translated_output = translator_en_zh(transcribed_input)
    print("Translated: ", translated_output)
    
    # Text to speech
    print("Speaking translated output...")
    zh_speaker.speak(translated_output)