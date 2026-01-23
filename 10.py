import json
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 한글 폰트 설정 (macOS)
# 시스템에 설치된 한글 폰트 찾기
def set_korean_font():
    # macOS에서 일반적으로 사용 가능한 한글 폰트 목록
    korean_fonts = ['AppleGothic', 'Apple SD Gothic Neo', 'NanumGothic', 'NanumBarunGothic', 
                    'Malgun Gothic', 'Arial Unicode MS']
    
    # 시스템 폰트 목록 가져오기
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 사용 가능한 한글 폰트 찾기
    for font in korean_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            print(f"한글 폰트 설정: {font}")
            return
    
    # 폰트를 찾지 못한 경우 경고만 출력
    print("⚠️ 한글 폰트를 찾지 못했습니다. 한글이 깨질 수 있습니다.")

set_korean_font()
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 파일 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, "eval_results_scored104_RR.json")
output_path = os.path.join(script_dir, "chapter_ookk_accuracy.png")

# 데이터 로드
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# chapter_ok 값 추출
chapter_ok_values = [item.get("chapter_ok", 0) for item in data if "chapter_ok" in item]

# 전체 정확도 계산 (chapter_ok=1 비율)
total_count = len(chapter_ok_values)
correct_count = sum(chapter_ok_values)
accuracy = correct_count / total_count if total_count > 0 else 0

print(f"전체 샘플 수: {total_count}")
print(f"정답 수 (chapter_ok=1): {correct_count}")
print(f"정확도 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")

# 0, 1 비율 계산
ratio_0 = (total_count - correct_count) / total_count
ratio_1 = correct_count / total_count

# 그래프 그리기
plt.figure(figsize=(6, 5))
data_for_plot = {
    "chapter_ok": ["0 (incorrect)", "1 (correct)"],
    "ratio": [ratio_0, ratio_1]
}

sns.barplot(data=data_for_plot, x="chapter_ok", y="ratio", palette=["#e74c3c", "#2ecc71"])
plt.ylim(0, 1)
plt.title(f"Chapter OK Accuracy\n(전체 정확도: {accuracy*100:.2f}%)", fontsize=14, fontweight="bold")
plt.xlabel("Chapter OK", fontsize=12)
plt.ylabel("Ratio", fontsize=12)

# 비율 값 표시
for i, (label, ratio) in enumerate(zip(data_for_plot["chapter_ok"], data_for_plot["ratio"])):
    plt.text(i, ratio + 0.02, f"{ratio:.3f}\n({ratio*100:.1f}%)", 
             ha="center", va="bottom", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()  # 그래프 창 닫기
print(f"\n그래프 저장 완료: {output_path}")
print(f"정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")