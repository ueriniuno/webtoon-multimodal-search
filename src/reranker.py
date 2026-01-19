# src/reranker.py
from sentence_transformers import CrossEncoder
from src.config import settings
from src.schemas import SearchResult

class Reranker:
    def __init__(self):
        model_id = settings.models['reranker']
        print(f"🧠 Reranker 모델 로딩 중: {model_id}")
        # GPU 자동 사용
        self.model = CrossEncoder(model_id, max_length=512)

    def rerank(self, query: str, docs: list[SearchResult], top_k: int) -> list[SearchResult]:
        """
        문서 리스트를 받아서 질문(query)과의 적합도 점수를 매기고 정렬하여 반환
        """
        if not docs:
            return []

        # Cross-Encoder 입력 생성: [[질문, 본문1], [질문, 본문2], ...]
        pairs = [[query, doc.full_context_text] for doc in docs]
        
        # 채점 실행 (Scores 리스트 반환)
        scores = self.model.predict(pairs)
        
        # 점수 업데이트
        for doc, score in zip(docs, scores):
            doc.score = float(score)
            
        # 점수 높은 순으로 정렬 (내림차순)
        sorted_docs = sorted(docs, key=lambda x: x.score, reverse=True)
        
        # 상위 K개만 자르기
        return sorted_docs[:top_k]