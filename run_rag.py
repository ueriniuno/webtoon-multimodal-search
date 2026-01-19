#src/run_rag.py
import sys
import time
from src import WebtoonDB, EmbeddingEngine, ExaoneLLM, RAGPipeline

# =========================================================
# 🎚️ [실험 설정]
# WINDOW_SIZE: 검색된 씬의 앞뒤 N개를 추가로 가져와 문맥을 보강함
# =========================================================
WINDOW_SIZE = 0  # 4컷 묶음이라 0이어도 충분하지만, 필요시 1로 변경
# =========================================================

def main():
    print("\n" + "="*60)
    print(f"🤖 [Webtoon AI] RAG 시스템 시작 (Window: {WINDOW_SIZE})")
    print("="*60)

    try:
        # 1. 컴포넌트 초기화 (Dependency Injection)
        db = WebtoonDB()
        embedder = EmbeddingEngine()
        llm = ExaoneLLM()
        
        # 2. 파이프라인 조립
        pipeline = RAGPipeline(db, embedder, llm)
        print("\n✅ 시스템 준비 완료!")
        
    except Exception as e:
        print(f"\n❌ 초기화 실패: {e}")
        print("팁: 데이터 적재(run_ingest.py)를 먼저 실행했는지 확인하세요.")
        return

    print("\n💡 종료하려면 'q'를 입력하세요.")
    
    while True:
        print("\n" + "-"*60)
        user_query = input("💬 질문: ").strip()
        
        if user_query.lower() in ['exit', 'quit', 'q']:
            print("👋 종료합니다.")
            break
        if not user_query: continue

        try:
            start_time = time.time()
            
            # 파이프라인 실행
            # (window_size 인자는 pipeline.run 메서드 확장이 필요하지만,
            # 현재 기본 pipeline.py는 이를 지원하지 않으므로 일단 query만 넘김)
            # 만약 window_size를 구현하고 싶다면 pipeline.py의 retrieve 부분 수정 필요
            # 상단에 설정한 WINDOW_SIZE 값을 전달
            answer = pipeline.run(user_query, window_size=WINDOW_SIZE)
            
            end_time = time.time()
            
            print("\n" + "="*60)
            print(f"🎯 [AI 답변] ({end_time - start_time:.2f}초)")
            print("="*60)
            print(answer)

        except Exception as e:
            print(f"⚠️ 에러 발생: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()