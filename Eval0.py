import os
import pandas as pd
import re
import json
from tqdm.auto import tqdm

tqdm.pandas()

# 1. 데이터 전처리
def preprocess_eval_data(json_data):
    df = pd.DataFrame(json_data)
    # Ground Truth에서 비교용 숫자(화수) 추출
    df['gt_chapter'] = df['ground_truth'].apply(
        lambda x: re.search(r'(\d+)', str(x)).group(1) if re.search(r'(\d+)', str(x)) else "0"
    )
    return df

# 2. 핵심 정보(화수) 포함 여부 자동 체크
def check_chapter_match(row):
    """
    답변 텍스트 내에 gt_chapter(숫자)가 포함되어 있는지 확인
    다양한 패턴을 지원하여 정확도를 높임
    """
    ans_text = str(row['model_response'])
    gt_num = str(row['gt_chapter']).strip()
    
    if not gt_num or not ans_text:
        return 0
    
    # 패턴 목록 (우선순위 순서)
    patterns = [
        # 1. 한국어 기본 패턴
        rf'{gt_num}\s*화',                    # "10화", "10 화"
        rf'\[{gt_num}\s*화\]',                # "[10화]", "[10 화]"
        rf'\({gt_num}\s*화\)',                # "(10화)"
        rf'화\s+{gt_num}',                    # "화 10" (역순)
        rf'\[화\s+{gt_num}',                  # "[화 10", "[화 10, 컷 5]"
        rf'{gt_num}\s*회',                    # "10회"
        rf'{gt_num}\s*편',                    # "10편"
        rf'에피소드\s+{gt_num}',              # "에피소드 10"
        rf'\[에피소드\s+{gt_num}\]',          # "[에피소드 10]"
        
        # 2. 영어 패턴 (대소문자 구분 없음)
        rf'EPISODE\s+{gt_num}',               # "EPISODE 10"
        rf'Episode\s+{gt_num}',               # "Episode 10"
        rf'episode\s+{gt_num}',               # "episode 10"
        rf'EPISODE\s+{gt_num}\s*[,{{]',      # "EPISODE 10,", "EPISODE 10 {"
        rf'EPISODE\s+{gt_num}\s*\(',         # "EPISODE 10 ("
        rf'EP\s+{gt_num}',                    # "EP 10"
        rf'Ep\s+{gt_num}',                    # "Ep 10"
        
        # 3. Ch. 패턴
        rf'\[Ch\.\s*{gt_num}\]',              # "[Ch.10]", "[Ch. 10]"
        rf'Ch\.\s*{gt_num}',                  # "Ch.10", "Ch. 10"
        rf'\[Ch\.{gt_num}\]',                 # "[Ch.10]" (공백 없음)
        
        # 4. CUT 패턴과 함께
        rf'EPISODE\s+{gt_num}\s*,\s*CUT',     # "EPISODE 10, CUT 82-83"
        rf'Episode\s+{gt_num}\s*,\s*CUT',    # "Episode 10, CUT"
        rf'EPISODE\s+{gt_num}\s*,\s*Cut',    # "EPISODE 10, Cut"
        
        # 5. 괄호 안에 화수 (예: "EPISODE 19 (19화)")
        rf'\({gt_num}\s*화\)',                # "(10화)"
        rf'EPISODE\s+{gt_num}\s*\({gt_num}',  # "EPISODE 19 (19화)"
        
        # 6. 숫자만 (단어 경계로 구분, 다른 숫자에 포함되지 않도록)
        # 주의: 이 패턴은 마지막에 와야 함 (더 구체적인 패턴이 우선)
        # 숫자 앞뒤에 공백, 구두점, 또는 단어 경계가 있는 경우
        rf'(?<!\d){gt_num}(?!\d)',           # 숫자만 "10" (다른 숫자에 포함되지 않음)
        rf'\b{gt_num}\b',                    # 단어 경계로 구분된 숫자 "10"
    ]
    
    # 각 패턴을 대소문자 구분 없이 검색
    for pattern in patterns:
        if re.search(pattern, ans_text, re.IGNORECASE):
            return 1
    
    return 0

# --- 실행 파트 ---

# 데이터 로드: eval_results.json에서 불러오기 (model_response 포함)
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, "eval_results_104_RR.json")
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df = preprocess_eval_data(data)
# model_response 컬럼이 있을 때만 평가 진행
if 'model_response' in df.columns:
    print(f"총 {len(df)}개 샘플에 대해 평가를 진행합니다.")

    # 화수 포함 여부 체크
    print("화수 포함 여부 체크 중...")
    df['chapter_ok'] = df.progress_apply(check_chapter_match, axis=1)

    # 결과 저장
    output_path = os.path.join(script_dir, "eval_results_scored104_RR.json")
    df.to_json(output_path, force_ascii=False, orient="records", indent=2)

    print("평가 완료! 결과가 다음 경로에 저장되었습니다:")
    print(output_path)
    print(f"\nchapter_ok 통계:")
    print(f"  - 정답 (1): {df['chapter_ok'].sum()}개")
    print(f"  - 오답 (0): {(df['chapter_ok'] == 0).sum()}개")
    print(f"  - 정확도: {df['chapter_ok'].mean():.2%}")
else:
    print("⚠️ 현재 데이터에는 'model_response' 컬럼이 없습니다. 질문·정답 정보만 로드했습니다.")