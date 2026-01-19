# src/embedding.py
from sentence_transformers import SentenceTransformer
from src.config import settings
import torch

class EmbeddingEngine:
    def __init__(self):
        model_id = settings.models["embedding"]
        # GPU 사용 가능 여부 확인
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 임베딩 모델 로드 중 ({device}): {model_id}")
        
        self.model = SentenceTransformer(model_id, device=device)

    def get_embeddings(self, text):
        """텍스트 -> 768차원 벡터 변환"""
        # normalize_embeddings=True를 쓰면 코사인 유사도 계산이 더 정확해짐
        return self.model.encode(text, normalize_embeddings=True)