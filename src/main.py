# python src/main.py

import cv2
import os
from dotenv import load_dotenv

from detector import run_detector
from ocr.clova import ClovaOCR

# env 로드
load_dotenv()

INVOKE_URL = os.getenv("CLOVA_OCR_INVOKE_URL")
SECRET_KEY = os.getenv("CLOVA_OCR_SECRET")

if not INVOKE_URL or not SECRET_KEY:
    raise RuntimeError("CLOVA OCR 환경변수가 설정되지 않았습니다")

# 테스트 이미지 경로
IMAGE_PATH = "images/sample.png"

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise RuntimeError("이미지를 불러올 수 없습니다")

# 1️⃣ 말풍선/텍스트 탐지
blocks = run_detector(IMAGE_PATH)

# 2️⃣ OCR 객체
ocr = ClovaOCR(
    invoke_url=INVOKE_URL,
    secret_key=SECRET_KEY
)

# 3️⃣ bbox → crop → OCR
for idx, block in enumerate(blocks):
    x1, y1, x2, y2 = block.xyxy

    # padding (웹툰에서 중요)
    pad = 8
    h, w, _ = image.shape
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    crop = image[y1:y2, x1:x2]

    texts = ocr.run(crop)

    print(f"\n🟦 Block {idx} | bbox={block.xyxy}")
    for t in texts:
        print("  -", t["text"], f"(conf={t['confidence']:.2f})")
