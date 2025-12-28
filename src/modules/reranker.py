# src/modules/reranker.py
from config import TOP_K_RERANK, USE_RERANKER
from model_loader import models

def rerank(query: str, retrieved_docs: list):
    if not USE_RERANKER or not retrieved_docs:
        return retrieved_docs[:TOP_K_RERANK]

    # (Query, Document Text) 쌍 생성
    pairs = []
    for doc in retrieved_docs:
        doc_text = f"{doc.payload['scene_summary']} {doc.payload['ocr_text']}"
        pairs.append([query, doc_text])

    # Cross-Encoder로 점수 계산
    scores = models.reranker.predict(pairs)

    # 점수 높은 순 정렬
    for doc, score in zip(retrieved_docs, scores):
        doc.score = score  # 점수 덮어쓰기

    # 정렬 및 상위 K개 추출
    ranked_docs = sorted(retrieved_docs, key=lambda x: x.score, reverse=True)
    return ranked_docs[:TOP_K_RERANK]