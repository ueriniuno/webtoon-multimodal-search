import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = "/Users/hyunseo/Library/Mobile Documents/com~apple~CloudDocs/STUDY/BOAZ/Eval"
RESULT_PATH = os.path.join(BASE_DIR, "eval_results_scoredR.json")
RAW_PATH = os.path.join(BASE_DIR, "eval_results_RR.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")


def load_results(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # 점수 타입 정리
    if "eval_score" in df.columns:
        df["eval_score"] = pd.to_numeric(df["eval_score"], errors="coerce")

    # question_difficulty가 scored 결과에 없으면 raw(RR)에서 id로 보강
    if "question_difficulty" not in df.columns and "id" in df.columns and os.path.exists(RAW_PATH):
        try:
            with open(RAW_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            raw_df = pd.DataFrame(raw)
            if "id" in raw_df.columns and "question_difficulty" in raw_df.columns:
                df = df.merge(
                    raw_df[["id", "question_difficulty"]],
                    on="id",
                    how="left",
                )
        except Exception as e:
            print(f"[WARN] question_difficulty 머지 실패: {e}")

    return df


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_score_hist(df: pd.DataFrame):
    plt.figure(figsize=(6, 4))
    sns.histplot(df["eval_score"].dropna(), bins=6, kde=False)
    plt.title("분포: 서술형 평가 점수")
    plt.xlabel("eval_score")
    plt.ylabel("개수")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "score_hist.png"))
    plt.close()


def plot_score_by_chapter(df: pd.DataFrame):
    if "gt_chapter" not in df.columns:
        return
    plt.figure(figsize=(8, 4))
    # 장별 평균 점수
    chap_mean = df.groupby("gt_chapter")["eval_score"].mean().reset_index()
    sns.barplot(data=chap_mean, x="gt_chapter", y="eval_score")
    plt.title("장(화수)별 평균 평가 점수")
    plt.xlabel("gt_chapter")
    plt.ylabel("평균 eval_score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "score_by_chapter.png"))
    plt.close()


def plot_score_vs_chapter_ok(df: pd.DataFrame):
    if "chapter_ok" not in df.columns:
        return
    plt.figure(figsize=(5, 4))
    sns.boxplot(
        data=df,
        x="chapter_ok",
        y="eval_score",
    )
    plt.title("화수 정합 여부(chapter_ok) vs 평가 점수")
    plt.xlabel("chapter_ok (0=불일치, 1=일치)")
    plt.ylabel("eval_score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "score_vs_chapter_ok.png"))
    plt.close()


def plot_chapter_ok_accuracy(df: pd.DataFrame):
    if "chapter_ok" not in df.columns:
        return

    # 전체 정확도 (chapter_ok=1 비율)
    accuracy = df["chapter_ok"].mean()

    # 0,1 비율 막대그래프
    plt.figure(figsize=(4, 4))
    acc_df = (
        df["chapter_ok"]
        .value_counts(normalize=True)
        .rename({0: "0 (incorrect)", 1: "1 (correct)"})
        .reset_index()
    )
    acc_df.columns = ["chapter_ok_label", "ratio"]

    sns.barplot(data=acc_df, x="chapter_ok_label", y="ratio")
    plt.ylim(0, 1)
    plt.title(f"chapter_ok standard accuracy)")
    plt.ylabel("ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "chapter_ok_accuracy.png"))
    plt.close()


def plot_accuracy_by_difficulty(df: pd.DataFrame):
    """
    question_difficulty(질문 난이도)별 정답률(= chapter_ok 평균)을 % 비율로 시각화
    """
    if "chapter_ok" not in df.columns or "question_difficulty" not in df.columns:
        return

    # 난이도를 숫자로 정렬 가능한 형태로 변환 (예: "1","2","3"...)
    diff_series = pd.to_numeric(df["question_difficulty"], errors="coerce")
    df = df.copy()
    df["question_difficulty_num"] = diff_series

    # 난이도별 정답률 (chapter_ok 평균)
    acc_df = (
        df.dropna(subset=["question_difficulty_num"])
        .groupby("question_difficulty_num")["chapter_ok"]
        .mean()
        .reset_index()
        .sort_values("question_difficulty_num")
    )
    # 비율(0~1)을 %로 변환
    acc_df["accuracy_percent"] = acc_df["chapter_ok"] * 100
    
    # 난이도 라벨 생성 (1=Easy, 5=Difficult)
    def get_difficulty_label(diff_num):
        if diff_num == 1:
            return "1 (Easy)"
        elif diff_num == 5:
            return "5 (Difficult)"
        else:
            return str(int(diff_num))
    
    acc_df["difficulty_label"] = acc_df["question_difficulty_num"].apply(get_difficulty_label)

    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=acc_df,
        x="difficulty_label",
        y="accuracy_percent",
        order=["1 (Easy)", "2", "3", "4", "5 (Difficult)"],
    )
    plt.ylim(0, 100)
    plt.title("question_difficulty by accuracy (%)")
    plt.xlabel("question_difficulty")
    plt.ylabel("accuracy (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_by_difficulty.png"))
    plt.close()


def plot_difficulty_ratio(df: pd.DataFrame):
    """
    question_difficulty 분포 자체(난이도별 샘플 비율)를 시각화
    """
    if "question_difficulty" not in df.columns:
        return

    diff_series = pd.to_numeric(df["question_difficulty"], errors="coerce")
    df = df.copy()
    df["question_difficulty_num"] = diff_series

    ratio_df = (
        df.dropna(subset=["question_difficulty_num"])
        ["question_difficulty_num"]
        .value_counts(normalize=True)
        .sort_index()
        .reset_index()
    )
    ratio_df.columns = ["question_difficulty_num", "ratio"]
    
    # 난이도 라벨 생성 (1=Easy, 5=Difficult)
    def get_difficulty_label(diff_num):
        if diff_num == 1:
            return "1 (Easy)"
        elif diff_num == 5:
            return "5 (Difficult)"
        else:
            return str(int(diff_num))
    
    ratio_df["difficulty_label"] = ratio_df["question_difficulty_num"].apply(get_difficulty_label)

    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=ratio_df,
        x="difficulty_label",
        y="ratio",
        order=["1 (Easy)", "2", "3", "4", "5 (Difficult)"],
    )
    plt.ylim(0, 1)
    plt.title("난이도(question_difficulty)별 샘플 비율")
    plt.xlabel("question_difficulty")
    plt.ylabel("비율")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "difficulty_ratio.png"))
    plt.close()


def main():
    ensure_output_dir()
    df = load_results(RESULT_PATH)

    print(f"데이터 로드 완료: {len(df)}개 샘플")

    plot_score_hist(df)
    plot_score_by_chapter(df)
    plot_score_vs_chapter_ok(df)
    plot_chapter_ok_accuracy(df)
    plot_accuracy_by_difficulty(df)
    plot_difficulty_ratio(df)

    print(f"그래프 저장 완료: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

