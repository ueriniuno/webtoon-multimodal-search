# python src/main.py

import cv2
import os
import json
import sys
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

from detector import run_detector
from ocr.clova import ClovaOCR

# UTF-8 출력 설정
sys.stdout.reconfigure(encoding='utf-8')

# env 로드
load_dotenv()

INVOKE_URL = os.getenv("CLOVA_OCR_INVOKE_URL")
SECRET_KEY = os.getenv("CLOVA_OCR_SECRET")

if not INVOKE_URL or not SECRET_KEY:
    raise RuntimeError("CLOVA OCR 환경변수가 설정되지 않았습니다")

# 자연스러운 정렬 함수
def natural_sort_key(path):
    """파일명을 자연스럽게 정렬 (Windows 탐색기처럼)"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(path.name))]

# 이미지 폴더 경로 (src 폴더 기준)
SRC_DIR = Path(__file__).parent
IMAGE_DIR = SRC_DIR / "images" / "total_processed"

# 경로 검증
if not IMAGE_DIR.exists():
    raise RuntimeError(f"❌ 폴더가 존재하지 않습니다: {IMAGE_DIR}\n"
                       f"   다음 경로에 이미지를 넣어주세요: {IMAGE_DIR.absolute()}")

print(f"✓ 폴더 확인: {IMAGE_DIR}")

# OCR 객체 생성 (재사용)
ocr = ClovaOCR(
    invoke_url=INVOKE_URL,
    secret_key=SECRET_KEY
)

# 이미지 파일 리스트 가져오기 (자연스러운 정렬)
all_image_files = sorted(IMAGE_DIR.glob("*.png"), key=natural_sort_key)

if not all_image_files:
    raise RuntimeError(f"{IMAGE_DIR}에 PNG 이미지가 없습니다")

# ⚙️ 처리할 이미지 개수 설정
TEST_LIMIT = None # 원하는 개수로 변경 (None이면 전체)

if TEST_LIMIT:
    image_files = all_image_files[:TEST_LIMIT]
    print(f"📁 전체 {len(all_image_files)}개 중 {len(image_files)}개만 처리 (테스트 모드)\n")
else:
    image_files = all_image_files
    print(f"📁 총 {len(image_files)}개 이미지 발견 (전체 모드)\n")

# 결과 저장용
results = []

# 각 이미지 처리
for img_idx, img_path in enumerate(tqdm(image_files, desc="🔍 OCR 처리 중"), start=1):
    image_result = {
        "image_number": img_idx,
        "filename": img_path.name,
        "status": "success",
        "blocks": []
    }
    
    # 이미지 로드 (PIL 사용 - 경로 문제 없음)
    try:
        from PIL import Image
        import numpy as np
        
        pil_img = Image.open(img_path)
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        image_result["status"] = "failed"
        image_result["error"] = f"이미지 로드 실패: {str(e)}"
        results.append(image_result)
        continue

    # 1️⃣ 말풍선/텍스트 탐지
    try:
        blocks = run_detector(str(img_path))
    except Exception as e:
        image_result["status"] = "failed"
        image_result["error"] = f"탐지 실패: {str(e)}"
        results.append(image_result)
        continue

    if not blocks:
        image_result["status"] = "no_blocks"
        results.append(image_result)
        continue

    # 2️⃣ bbox → crop → OCR
    for block_idx, block in enumerate(blocks):
        x1, y1, x2, y2 = block.xyxy

        # padding
        pad = 8
        h, w, _ = image.shape
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop = image[y1:y2, x1:x2]

        block_result = {
            "block_number": block_idx,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "texts": []
        }

        try:
            texts = ocr.run(crop)
            block_result["texts"] = texts
        except Exception as e:
            block_result["error"] = str(e)

        image_result["blocks"].append(block_result)
    
    results.append(image_result)

# 결과 저장 (src 폴더에 저장)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f"ocr_results_{timestamp}.json"
output_path = SRC_DIR / output_file

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n\n✅ 전체 처리 완료!")
print(f"📊 총 {len(image_files)}개 이미지 처리됨")
print(f"💾 결과 저장: {output_path}")

# 간단한 통계 출력
success_count = sum(1 for r in results if r["status"] == "success")
failed_count = sum(1 for r in results if r["status"] == "failed")
no_blocks_count = sum(1 for r in results if r["status"] == "no_blocks")

print(f"\n📈 통계:")
print(f"   ✓ 성공: {success_count}개")
print(f"   ✗ 실패: {failed_count}개")
print(f"   ○ 텍스트 없음: {no_blocks_count}개")