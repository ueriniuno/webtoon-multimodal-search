from src.database import WebtoonDB
from src.embedding import EmbeddingEngine
from src.models import QwenLLM
from src.pipeline import RAGPipeline
import sys

def main():
    # 1. 시스템 초기화
    db = WebtoonDB("./qdrant_storage")
    embedder = EmbeddingEngine()
    llm = QwenLLM()
    pipeline = RAGPipeline(db, embedder, llm)

    print("\n" + "="*50)
    print("🎨 웹툰 장면 검색 RAG 시스템 (Direct Search Mode)")
    print("설명: 리라이터 없이 사용자의 질문으로 직접 검색합니다.")
    print("="*50)

    while True:
        user_query = input("\n💬 질문을 입력하세요: ").strip()

        if user_query.lower() in ['exit', 'quit', '종료', 'q']:
            print("👋 시스템을 종료합니다.")
            break

        if not user_query: continue

        try:
            # [수정] Rewriter를 거치지 않고 원본 질문(user_query)을 그대로 사용
            print(f"🔍 [Direct Search]: '{user_query}'로 검색 중...")
            
            # 2. 정보 검색 (Retriever) - 원본 질문 사용
            docs = pipeline.retrieve(user_query)
            
            if docs:
                # 3. 문서 선택 (Reranker) - 원본 질문 사용
                best_doc = pipeline.rerank(user_query, docs)
                
                # 참조 과정 출력
                print("\n📂 [AI가 참조한 원본 데이터]")
                source_file = best_doc.payload.get('image_file', '파일명 정보 없음')
                print(f"📍 참조 파일명: {source_file}")
                print(f"📝 원본 캡션: {best_doc.payload['full_text'][:200]}...")

                # 4. 최종 답변 생성 (Generator)
                answer = pipeline.generate_answer(user_query, best_doc.payload['full_text'])
                
                print("\n" + "—"*50)
                print(f"🎯 AI의 최종 답변:\n{answer}")
                print("—"*50)
            else:
                print("❌ 관련 정보를 찾을 수 없습니다.")

        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")

if __name__ == "__main__":
    main()