from sentence_transformers import SentenceTransformer

class EmbeddingEngine:
    def __init__(self, model_id='BAAI/bge-m3'):
        print(f"🚀 임베딩 모델 로드 중: {model_id}")
        self.model = SentenceTransformer(model_id)

    def get_embeddings(self, text):
        return self.model.encode(text, normalize_embeddings=True)