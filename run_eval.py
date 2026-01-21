import json
import time
import os
import sys

# 프로젝트 루트 경로 추가 (src 폴더 인식용)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 👇 사용자의 실제 모듈 임포트
from src import WebtoonDB, EmbeddingEngine, ExaoneLLM, RAGPipeline

# =========================================================
# 1. 설정 및 초기화
# =========================================================
WINDOW_SIZE = 0  # run_rag.py 설정과 동일하게 맞춤

def load_eval_data(file_path):
    """평가 데이터셋 로드"""
    if not os.path.exists(file_path):
        print(f"❌ 데이터 파일이 없습니다: {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"📂 데이터 로드 완료: {len(data)}개 항목")
    return data

def initialize_pipeline():
    """파이프라인 컴포넌트 초기화 및 조립"""
    print("⚙️ [Setup] 평가용 파이프라인 컴포넌트 초기화 중...")
    
    try:
        # run_rag.py와 동일한 방식으로 컴포넌트 생성
        db = WebtoonDB()
        embedder = EmbeddingEngine()
        llm = ExaoneLLM()
        
        # 파이프라인 생성
        pipeline = RAGPipeline(db, embedder, llm)
        print("✅ 파이프라인 조립 완료!")
        return pipeline
        
    except Exception as e:
        print(f"❌ 파이프라인 초기화 실패: {e}")
        return None

# =========================================================
# 2. 평가 실행 로직
# =========================================================
def run_evaluation(pipeline, data):
    results = []
    total = len(data)
    
    print(f"\n🚀 총 {total}개의 질문에 대한 평가를 시작합니다.\n")
    
    for idx, item in enumerate(data):
        qid = item.get('id', idx + 1)
        question = item['question']
        ground_truth = item.get('ground_truth', "N/A")
        
        print(f"--- [{idx+1}/{total}] ID: {qid} ---")
        print(f"❓ 질문: {question}")
        
        start_time = time.time()
        try:
            # 실제 RAG 파이프라인 실행
            # (이 과정에서 pipeline.py 내부의 _save_debug_log가 실행되어 debug_search_log.txt에 기록됩니다)
            response = pipeline.run(question, window_size=WINDOW_SIZE)
            
        except Exception as e:
            print(f"❌ 에러 발생: {e}")
            response = f"[Error] {str(e)}"
            time.sleep(2)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # [화면 출력용] 너무 길면 화면이 지저분해지니 앞부분만 보여줍니다.
        preview = response.replace("\n", " ")[:80]
        print(f"🤖 답변: {preview}...")
        print(f"⏱ 소요: {elapsed:.2f}초\n")
        
        # [파일 저장용] 여기에는 'response' 전체 원본을 저장합니다. (요약 X)
        results.append({
            "id": qid,
            "question": question,
            "model_response": response,  # 👈 전체 답변 저장됨
            "ground_truth": ground_truth,
            "latency_seconds": round(elapsed, 2)
        })
        
        # 시스템 과부하 방지
        time.sleep(0.5)
        
    return results

# =========================================================
# 3. 메인 실행부
# =========================================================
if __name__ == "__main__":
    # 파일명 설정
    input_file = 'eval_data.json'
    output_file = 'eval_results.json'
    
    # 1. 평가 데이터 로드
    eval_list = load_eval_data(input_file)
    
    if eval_list:
        # 2. 파이프라인 준비
        rag_pipeline = initialize_pipeline()
        
        if rag_pipeline:
            # 3. 평가 수행
            final_results = run_evaluation(rag_pipeline, eval_list)
            
            # 4. 결과 파일 저장
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)
                
            print(f"\n🎉 모든 평가가 완료되었습니다!")
            print(f"📄 결과 파일(전체 답변 포함) 저장됨: {os.path.abspath(output_file)}")
            print(f"📝 검색 로그 저장됨: debug_search_log.txt (누적됨)")
        else:
            print("⚠️ 파이프라인이 생성되지 않아 평가를 중단합니다.")
    else:
        print("⚠️ 평가할 데이터가 없어 종료합니다.")