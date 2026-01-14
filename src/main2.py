# python src/main2.py

import cv2
import os
import json
import sys
import re
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from detector import run_detector
from ocr.clova import ClovaOCR

# UTF-8 출력 설정
sys.stdout.reconfigure(encoding="utf-8")

# env 로드
load_dotenv()

INVOKE_URL = os.getenv("CLOVA_OCR_INVOKE_URL")
SECRET_KEY = os.getenv("CLOVA_OCR_SECRET")

if not INVOKE_URL or not SECRET_KEY:
    raise RuntimeError("CLOVA OCR 환경변수가 설정되지 않았습니다")

# 자연 정렬
def natural_sort_key(path):
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", path.name)
    ]

# =========================
# 경로 설정
# =========================
SRC_DIR = Path(__file__).parent

IMAGE_ROOT = SRC_DIR / "im"              # 🔹 1~7 화 폴더
OCR_OUTPUT_ROOT = SRC_DIR / "json_data_ocr_cut_seg"

OCR_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

if not IMAGE_ROOT.exists():
    raise RuntimeError(f"❌ 이미지 루트 폴더 없음: {IMAGE_ROOT}")

print(f"✓ 이미지 루트: {IMAGE_ROOT}")
print(f"✓ OCR 결과 루트: {OCR_OUTPUT_ROOT}")

# OCR 객체
ocr = ClovaOCR(
    invoke_url=INVOKE_URL,
    secret_key=SECRET_KEY
)

# =========================
# 화별 OCR 처리
# =========================
episode_dirs = sorted(
    [d for d in IMAGE_ROOT.iterdir() if d.is_dir()],
    key=lambda x: int(x.name)
)

print(f"\n📘 총 {len(episode_dirs)}화 처리 시작\n")

total_success = 0
total_failed = 0
total_no_blocks = 0

for ep_dir in episode_dirs:
    ep = ep_dir.name
    print(f"\n▶ {ep}화 처리 중...")

    output_dir = OCR_OUTPUT_ROOT / ep
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(ep_dir.glob("*.png"), key=natural_sort_key)

    if not image_files:
        print(f"⚠ {ep}화: 이미지 없음")
        continue

    for img_path in tqdm(image_files, desc=f"🔍 {ep}화 OCR"):
        image_file = img_path.name

        # 이미지 로드
        try:
            from PIL import Image
            import numpy as np

            pil_img = Image.open(img_path)
            image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            total_failed += 1
            continue

        # 텍스트 블록 탐지
        try:
            blocks = run_detector(str(img_path))
        except Exception:
            total_failed += 1
            continue

        if not blocks:
            total_no_blocks += 1
            continue

        ocr_blocks = []

        for block_idx, block in enumerate(blocks):
            x1, y1, x2, y2 = block.xyxy

            pad = 8
            h, w, _ = image.shape
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            crop = image[y1:y2, x1:x2]

            try:
                texts = ocr.run(crop)
            except Exception:
                continue

            merged_texts = [
                t["text"].strip()
                for t in texts
                if t.get("text")
            ]

            if not merged_texts:
                continue

            ocr_blocks.append({
                "block_number": block_idx,
                "text": " ".join(merged_texts)
            })

        # 이미지 1장 = JSON 1개
        out_json = {
            "episode": ep,
            "image_file": image_file,
            "ocr": ocr_blocks
        }

        out_path = output_dir / f"{img_path.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)

        total_success += 1

# =========================
# 요약
# =========================
print("\n✅ 전체 OCR 완료")
print(f"   ✓ 성공: {total_success}")
print(f"   ✗ 실패: {total_failed}")
print(f"   ○ 텍스트 없음: {total_no_blocks}")
print(f"📂 결과 위치: {OCR_OUTPUT_ROOT}")
