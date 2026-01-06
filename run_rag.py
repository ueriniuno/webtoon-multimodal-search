from src.database import WebtoonDB
from src.embedding import EmbeddingEngine
from src.models import QwenLLM
from src.pipeline import RAGPipeline
import sys

def main():
    # 1. 시스템 초기화 (모델 및 DB 연결)
    db = WebtoonDB("./qdrant_storage")
    embedder = EmbeddingEngine()
    llm = QwenLLM()
    pipeline = RAGPipeline(db, embedder, llm)

    print("\n" + "="*50)
    print("🎨 웹툰 장면 검색 RAG 시스템에 오신 것을 환영합니다!")
    print("종료하시려면 'exit' 또는 'quit'를 입력하세요.")
    print("="*50)

    while True:
        # 2. 사용자로부터 질문 입력 받기
        user_query = input("\n💬 질문을 입력하세요: ").strip()

        # 종료 조건 체크
        if user_query.lower() in ['exit', 'quit', '종료', 'q']:
            print("👋 시스템을 종료합니다. 감사합니다!")
            break

        if not user_query:
            continue

        try:
            # 3. RAG 파이프라인 가동
            # Rewriter
            refined_q = pipeline.rewrite_query(user_query)
            
            # Retriever
            docs = pipeline.retrieve(refined_q)
            
            if docs:
                # Reranker (현재는 최상위 1개 선택)
                best_doc = pipeline.rerank(refined_q, docs)
                
                # Generator
                answer = pipeline.generate_answer(user_query, best_doc.payload['full_text'])
                
                print("\n" + "—"*50)
                print(f"🎯 답변:\n{answer}")
                print("—"*50)
            else:
                print("❌ 관련 정보를 찾을 수 없습니다. 데이터를 확인해 주세요.")

        except Exception as e:
            print(f"⚠️ 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()