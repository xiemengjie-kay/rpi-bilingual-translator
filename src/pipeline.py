import f_whisper_button                             # Real-time speech-to-text using Faster-Whisper
import text_to_text_opus_en_zh as t2t_en_zh  # Text English-to-Chinese translation
import text_to_text_opus_zh_en as t2t_zh_en  # Text Chinese-to-English translation
import text_to_speech as tts                 # Text-to-speech using Piper-ONNX
import time
import signal # For graceful exit
import sys
import queue 

# --- Configuration for f_whisper_button ---
ENGLISH_BUTTON_PIN = 17
CHINESE_BUTTON_PIN = 27
WHISPER_MODEL_SIZE = "small" # default: base, consider "small" or "medium" for better speech recognition
WHISPER_COMPUTE_TYPE = "int8" # Recommended for RPi

if __name__ == "__main__":
    print("Initializing components...")
    fw = f_whisper_button.SpeechToTextSystem(
        english_button_pin=ENGLISH_BUTTON_PIN,
        chinese_button_pin=CHINESE_BUTTON_PIN,
        beam_size=5,
        model_size=WHISPER_MODEL_SIZE,
        compute_type=WHISPER_COMPUTE_TYPE
    )
    translator_en_zh = t2t_en_zh.EnglishToChineseTranslator()
    translator_zh_en = t2t_zh_en.ChineseToEnglishTranslator()
    zh_speaker = tts.TextToSpeech(voice_path="./piper_models/zh_CN-huayan-medium.onnx",
                                  config_path="./piper_models/zh_CN-huayan-medium.onnx.json")
    en_speaker = tts.TextToSpeech(voice_path="./piper_models/en_US-amy-medium.onnx",
                                  config_path="./piper_models/en_US-amy-medium.onnx.json")
    print("Components initialized. Ready for interaction.")

    # Flag to control the main loop's execution
    running = True

    # Signal handler for graceful exit (Ctrl+C)
    def signal_handler(sig, frame):
        print("\nCtrl+C detected. Shutting down...")
        global running
        running = False
        # You might need to add specific cleanup here for sounddevice or other resources if they don't exit cleanly
        # For gpiozero, it handles cleanup automatically when the script exits.
        # sounddevice streams are closed by f_whisper_button.SpeechToTextSystem._button_released_callback
        # but if you Ctrl+C while recording, you might need extra handling.
        # A simple exit might suffice for now.
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print(f"Start talking by pressing a button (GPIO {ENGLISH_BUTTON_PIN} for English, \
        GPIO {CHINESE_BUTTON_PIN} for Mandarin).")
    print("Press Ctrl+C to exit.")

    while running:
        try:
            # Try to get a result from the queue without blocking indefinitely
            # timeout=0.1 means it will wait for 100ms. If nothing, it raises queue.Empty
            transcription_data = fw.transcription_results_queue.get(timeout=0.1)

            transcribed_input = transcription_data['text']
            detected_language = transcription_data['language'] # 'en' or 'zh'

            if transcribed_input:
                print(f"Transcribed ({detected_language}): {transcribed_input}")

                if detected_language == "en": # English was spoken
                    # English to Chinese translation
                    translated_output = translator_en_zh(transcribed_input)
                    print(f"Translated to Chinese: {translated_output}")

                    # Text to speech in Chinese
                    print("Speaking translated Chinese output...")
                    zh_speaker.speak(translated_output)
                elif detected_language == "zh": # Chinese was spoken
                    # Chinese to English
                    translated_output = translator_zh_en(transcribed_input)
                    print(f"Translated to English: {translated_output}")

                    # Text to speech in English
                    print("Speaking translated English output...")
                    en_speaker.speak(translated_output)
            
            # Mark the task as done for the queue (important if you have many items)
            fw.transcription_results_queue.task_done()

        except queue.Empty:
            # No new transcription result, just continue the loop
            pass
        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}")
            # Consider if you want to exit or just log and continue
            time.sleep(1) # Prevent busy-waiting in case of continuous errors

    print("Main translation system gracefully shut down.")