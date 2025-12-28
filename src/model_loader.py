# src/model_loader.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import SentenceTransformer, CrossEncoder
from config import TEXT_EMBEDDING_MODEL, IMAGE_EMBEDDING_MODEL, RERANKER_MODEL, USE_RERANKER

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.init_models()
        return cls._instance

    def init_models(self):
        print("⏳ Loading Embedding Models...")
        self.text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
        self.image_model = SentenceTransformer(IMAGE_EMBEDDING_MODEL)
        
        # 차원 자동 계산
        self.text_dim = self.text_model.get_sentence_embedding_dimension()
        self.image_dim = self.image_model.get_sentence_embedding_dimension()

        if USE_RERANKER:
            print(f"⏳ Loading Reranker Model ({RERANKER_MODEL})...")
            self.reranker = CrossEncoder(RERANKER_MODEL)
        else:
            self.reranker = None
            
        print("✅ All Models Loaded.")

# 전역 인스턴스
models = ModelManager()