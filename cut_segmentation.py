import cv2
import numpy as np
import glob
import os
import math
import re
from collections import defaultdict


def natural_sort_key(filename):
    # 파일명 끝의 '-2', '-3' 숫자를 찾습니다.
    # 만약 숫자가 없다면(첫 번째 장), 1번으로 간주합니다.
    match = re.search(r'-(\d+)\.(png|jpg|jpeg)$', filename)
    if match:
        return int(match.group(1))
    return 1  # 숫자가 없는 파일(첫 장)은 1순위


def extract_episode_id(filename):
    # 날짜-시간 패턴(YYYY-MM-DD-HH_MM_SS)을 추출합니다.
    match = re.search(r'\d{4}-\d{2}-\d{2}-\d{2}_\d{2}_\d{2}', filename)
    return match.group() if match else None


def process_and_split_by_ratio(img, base_name, output_folder, bridge_size=100, ratio_threshold=1.2):
    if img is None: return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]

    edge_pixels = np.concatenate([gray[:5, :], gray[-5:, :], gray[:, :5].T, gray[:, -5:].T], axis=None)
    bg_brightness = np.median(edge_pixels)
    if bg_brightness > 128:
        _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    else:
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    dilated = cv2.dilate(thresh, np.ones((bridge_size, 1), np.uint8), iterations=1)
    horizontal_sum = np.sum(dilated, axis=1)

    panels = []
    is_panel, start_y = False, 0
    for y in range(height):
        if horizontal_sum[y] > 0 and not is_panel:
            start_y, is_panel = y, True
        elif horizontal_sum[y] == 0 and is_panel:
            if y - start_y > 50:
                panels.append((max(0, start_y - 10), min(height, y + 10)))
            is_panel = False
    if is_panel: panels.append((start_y, height))

    # 비율 기반 저장
    final_count = 0
    for top, bottom in panels:
        panel_h = bottom - top
        ratio = panel_h / width
        if ratio >= ratio_threshold:
            num = math.ceil(ratio)
            step = panel_h / num
            for i in range(num):
                st, sb = int(top + (i * step)), int(top + ((i + 1) * step))
                cv2.imwrite(os.path.join(output_folder, f"{base_name}_cut_{final_count:03d}.png"),
                            img[st:min(sb, bottom), :])
                final_count += 1
        else:
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_cut_{final_count:03d}.png"), img[top:bottom, :])
            final_count += 1
    return final_count


def run_webtoon_pipeline(input_dir, output_root):
    # 모든 이미지 가져오기
    all_files = glob.glob(os.path.join(input_dir, '*.*'))
    if not all_files:
        print("이미지 파일을 찾을 수 없습니다.")
        return

    # ID별로 그룹화
    episode_groups = defaultdict(list)
    for f in all_files:
        ep_id = extract_episode_id(os.path.basename(f))
        if ep_id: episode_groups[ep_id].append(f)

    # ★ 수정 포인트: 에피소드 ID(시간순)를 정렬하여 순번(1, 2, 3...) 부여
    sorted_ep_ids = sorted(episode_groups.keys())
    print(f"총 {len(sorted_ep_ids)}개의 화(Episode)를 감지했습니다.")

    for idx, ep_id in enumerate(sorted_ep_ids, 1):
        batch = episode_groups[ep_id]
        # 각 그룹 내에서 파일을 숫자 순서대로 정렬
        batch.sort(key=natural_sort_key)

        print(f"\n[작업 시작] {idx}화 (ID: {ep_id}, 파일 {len(batch)}개) 처리 중...")

        # 폴더명을 'episode_날짜' 대신 '1', '2', '3'으로 설정
        episode_out_dir = os.path.join(output_root, str(idx))
        if not os.path.exists(episode_out_dir):
            os.makedirs(episode_out_dir)

        # 이미지 로드 및 가로 너비 통일
        imgs = [cv2.imread(f) for f in batch]
        imgs = [img for img in imgs if img is not None]
        if not imgs: continue

        target_w = imgs[0].shape[1]
        resized = [cv2.resize(img, (target_w, int(img.shape[0] * (target_w / img.shape[1])))) for img in imgs]

        # 세로로 합치기
        merged_img = cv2.vconcat(resized)

        # 결과 저장 (파일명도 숫자에 맞춰 변경 가능)
        total_cuts = process_and_split_by_ratio(merged_img, f"ep_{idx}", episode_out_dir)
        print(f" -> {idx}화 성공: {total_cuts}개의 최종 컷을 저장했습니다.")


if __name__ == "__main__":
    input_path = 'im'
    output_path = './webtoon_final_cuts'

    run_webtoon_pipeline(input_path, output_path)
