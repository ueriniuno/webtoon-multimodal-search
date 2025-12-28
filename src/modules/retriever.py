# src/modules/retriever.py
from qdrant_client import QdrantClient
from config import QDRANT_PATH, COLLECTION_NAME, TOP_K_RETRIEVAL
from model_loader import models

client = QdrantClient(path=QDRANT_PATH)

def retrieve(query: str):
    # 텍스트 벡터 변환
    query_vec = models.text_model.encode(query).tolist()
    
    # Qdrant 검색
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        using="text", # 일단 텍스트 기반 검색 (이미지 검색 필요시 수정)
        limit=TOP_K_RETRIEVAL,
        with_payload=True
    )
    return hits.points