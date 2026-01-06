import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenLLM:
    def __init__(self, model_id="Qwen/Qwen2.5-1.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

    def ask(self, prompt):
        messages = [{"role": "system", "content": "웹툰 전문가 어시스턴트입니다."}, {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        ids = self.model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=512)
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].split("assistant\n")[-1].strip()