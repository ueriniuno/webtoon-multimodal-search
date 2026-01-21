import os
import pandas as pd
import re
import json
from tqdm.auto import tqdm
from openai import OpenAI

tqdm.pandas()

# OpenAI API 설정
# - 터미널에서: export OPENAI_API_KEY="..."
# - 모델: gpt-4o-mini (흔히 "GPT-4 미니"로 부름)
OPENAI_MODEL = "gpt-4o-mini"
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

# 1. 데이터 전처리
def preprocess_eval_data(json_data):
    df = pd.DataFrame(json_data)
    # Ground Truth에서 비교용 숫자(화수) 추출
    df['gt_chapter'] = df['ground_truth'].apply(
        lambda x: re.search(r'(\d+)', str(x)).group(1) if re.search(r'(\d+)', str(x)) else "0"
    )
    return df

# 2. 서술형 응답 정성 평가 (GPT 활용)
def get_semantic_score(row):
    """
    서술형 답변의 내용이 질문과 정답의 맥락에 얼마나 부합하는지 0~5점 척도로 평가합니다.
    """
    prompt = f"""
    당신은 웹툰 데이터베이스 기반 RAG 시스템의 답변 심사위원입니다.
    사용자의 질문에 대해 시스템이 생성한 '서술형 답변'이 실제 정답과 얼마나 일치하는지 평가하세요.

    [질문]: {row['question']}
    [실제 정답]: {row['ground_truth']}
    [시스템 답변]: {row['model_response']}

    평가 가이드라인:
    - 5점: 답변에 언급된 '화수'가 정확하며, 장면 묘사가 질문의 의도와 완벽히 부합함.
    - 3점: 화수는 맞으나 묘사가 부실하거나, 화수는 틀렸지만 질문에 해당하는 장면을 매우 정확하게 묘사함.
    - 1점: 화수도 틀리고 장면 묘사도 질문과 관련이 적음.
    - 0점: 전혀 관계없는 답변이거나 '데이터를 찾을 수 없다'는 식의 답변.

    반드시 아래 JSON 형식으로만 응답하세요:
    {{"score": 점수, "reason": "이유"}}
    """
    
    try:
        if not os.environ.get("OPENAI_API_KEY"):
            return pd.Series([0, "평가 실패(OPENAI_API_KEY 누락)"])

        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "너는 평가 심사위원이다. 반드시 JSON만 출력한다."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        res_text = (resp.choices[0].message.content or "").strip()
        result = json.loads(res_text)
        return pd.Series([result.get("score", 0), result.get("reason", "")])

    except Exception as e:
        print(f"[OPENAI ERROR] id={row.get('id', None)} error={e}")
        return pd.Series([0, "평가 실패(OpenAI 오류)"])

# 3. 핵심 정보(화수) 포함 여부 자동 체크
def check_chapter_match(row):
    # 답변 텍스트 내에 gt_chapter(숫자)가 포함되어 있는지 확인
    ans_text = str(row['model_response'])
    gt_num = row['gt_chapter']
    
    # "N화" 또는 "N 화" 또는 "[N화]" 형태 검색
    is_match = 1 if re.search(rf'{gt_num}\s*화', ans_text) else 0
    return is_match

# --- 실행 파트 ---

# 데이터 로드: eval_results.json에서 불러오기 (model_response 포함)
with open("/Users/hyunseo/Library/Mobile Documents/com~apple~CloudDocs/STUDY/BOAZ/Eval/eval_results_RR.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = preprocess_eval_data(data)
ß
# model_response 컬럼이 있을 때만 평가 진행
if 'model_response' in df.columns:
    print(f"총 {len(df)}개 샘플에 대해 평가를 진행합니다.")

    # 1단계: 자동 화수 일치 확인
    print("1단계: 화수 포함 여부 체크 중...")
    df['chapter_ok'] = df.progress_apply(check_chapter_match, axis=1)

    # 2단계: 서술형 정성 평가 (샘플링 실행 또는 전체 실행)
    # API 호출이므로 상위 5~10개 먼저 테스트 권장
    print("2단계: 서술형 정성 평가(Gemini 호출) 진행 중...")
    df[['eval_score', 'eval_reason']] = df.progress_apply(get_semantic_score, axis=1)

    # 결과 저장
    output_path = "/Users/hyunseo/Library/Mobile Documents/com~apple~CloudDocs/STUDY/BOAZ/Eval/eval_results_scoredR.json"
    df.to_json(output_path, force_ascii=False, orient="records", indent=2)

    print("평가 완료! 결과가 다음 경로에 저장되었습니다:")
    print(output_path)
else:
    print("⚠️ 현재 데이터에는 'model_response' 컬럼이 없습니다. 질문·정답 정보만 로드했습니다.")