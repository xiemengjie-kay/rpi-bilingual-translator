from transformers import MarianMTModel, MarianTokenizer
import time

class ChineseToEnglishTranslator:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-zh-en", num_beams=1):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.num_beams = num_beams

    def __call__(self, chinese_text):
        start_time = time.time()
        inputs = self.tokenizer(chinese_text, return_tensors="pt", padding=True)
        translated = self.model.generate(**inputs, num_beams=self.num_beams)
        res = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        end_time = time.time()
        # print(res)
        # print(f"Inference time: {end_time - start_time:.2f}s")
        return ' '.join([text for text in res])

if __name__ == "__main__":
    translator = ChineseToEnglishTranslator(num_beams=10) # 使用10个束

    chinese_text = [
        "语言是人类社会最重要的交际工具，对语言能力的培养和运用更是伴随着人生发展的各个阶段。",
        #"在诸多语言能力中，写作既是语言学习的核心环节，也是最为高阶和复杂的语言能力，涉及对文字的识记、理解、运用和表达等多个要素。"
    ]
    translated_text = translator(chinese_text)
    print(f"Original (Chinese): {chinese_text[0]}")
    print(f"Translated (with {translator.num_beams} beams): {translated_text}")