from transformers import MarianMTModel, MarianTokenizer
import time

class EnglishToChineseTranslator:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-zh", num_beams=1): # Added num_beams parameter
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.num_beams = num_beams # Store num_beams as an instance variable

    def __call__(self, english_text):
        start_time = time.time()
        inputs = self.tokenizer(english_text, return_tensors="pt", padding=True)
        # Pass num_beams to the generate method
        translated = self.model.generate(**inputs, num_beams=self.num_beams)
        res = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        end_time = time.time()
        # print(res)
        # print(f"Inference time: {end_time - start_time:.2f}s")
        return ' '.join([text for text in res])

# Example usage:
if __name__ == "__main__":
    translator = EnglishToChineseTranslator(num_beams=5) # Using 10 beams

    english_text = (
        "This man has spent 25 years leaving with the disastrous mistakes of his past. "
        "which has made him an exile in his hometown. and cost him his dearest relationships."
    )
    translated_text = translator(english_text)
    print(f"Original: {english_text}")
    print(f"Translated (with {translator.num_beams} beams): {translated_text}")