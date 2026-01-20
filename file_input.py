import json
import time

# 1. JSON 파일 로드 (파일명: eval_data.json)
def load_eval_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 2. 모델에게 질문을 던지는 가상의 함수 (실제 API나 함수로 대체)
def get_model_response(question):
    # 여기에 실제 RAG 시스템이나 LLM 호출 로직이 들어갑니다.
    print(f"입력된 질문: {question}")
    return "모델의 답변 결과..."

# 3. 메인 실행부: 1~50번까지 순회하며 입력
def run_evaluation(data):
    results = []
    
    for item in data:
        print(f"--- [ID: {item['id']}] 테스트 시작 ---")
        
        # 실제 모델에 입력
        response = get_model_response(item['question'])
        
        # 결과 기록 (평가를 위해 정답과 함께 저장)
        results.append({
            "id": item["id"],
            "question": item["question"],
            "model_response": response,
            "ground_truth": item["ground_truth"]
        })
        
        # 시스템 과부하 방지나 시연을 위한 짧은 대기 (선택 사항)
        time.sleep(0.5) 
        
    return results

# 실행 예시
if __name__ == "__main__":
    # 데이터 로드
    eval_list = load_eval_data('eval_data.json')
    
    # 평가 실행
    final_results = run_evaluation(eval_list)
    
    # 결과를 다시 파일로 저장
    with open('eval_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
        
    print("\n모든 질문 입력 및 결과 저장이 완료되었습니다.")