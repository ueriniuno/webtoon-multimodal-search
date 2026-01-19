# src/models.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import settings

class ExaoneLLM:
    def __init__(self):
        model_id = settings.models["llm"]
        print(f"🚀 LLM 로드 중: {model_id}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # 모델 로드 (A100 최적화: bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    def ask(self, system_message, user_message):
        """
        채팅 템플릿을 적용하여 답변 생성
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Prompt 템플릿 적용
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 토큰화 및 GPU 이동
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        
        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,      # 답변 최대 길이
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,           # 창의적인 답변 허용
                temperature=0.7,          # 다양성 조절
                top_p=0.9
            )
            
        # 디코딩 (입력 프롬프트 제외하고 답변만 추출)
        input_length = inputs.input_ids.shape[1]
        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        return response.strip()