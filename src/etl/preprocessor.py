import os
import json
import re  # 정규표현식 모듈 추가 (파일명 분석용)
import torch
import easyocr
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from config import IMAGE_DIR, METADATA_FILE

def run_preprocessing():
    # 1. 이미지 폴더 확인
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ '{IMAGE_DIR}' 폴더가 없습니다.")
        return

    print("🚀 Starting Pre-processing (OCR + Captioning)...")
    
    # 2. 모델 로드 (OCR & VLM)
    print("⏳ Loading AI Models...")
    reader = easyocr.Reader(['ko', 'en']) # 한국어, 영어 인식
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # GPU 가속 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ Models Loaded on {device}")

    # 3. 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    dataset = []

    print(f"📂 Found {len(image_files)} images. Processing start!")

    # 4. 반복문 시작 (이미지 하나씩 처리)
    for img_file in tqdm(image_files):
        try:
            img_path = os.path.join(IMAGE_DIR, img_file)
            
            # --- [핵심 추가 기능] 파일명에서 에피소드/컷 정보 추출 ---
            # 규칙: ep숫자_숫자 (예: ep01_024.jpg -> 1화, 24컷)
            episode = -1
            scene_no = -1
            
            # 정규식: "ep" 뒤에 숫자 + "_" + 숫자 패턴 찾기
            match = re.search(r'ep(\d+)[_](\d+)', img_file.lower())
            if match:
                episode = int(match.group(1))
                scene_no = int(match.group(2))
            # ----------------------------------------------------

            # 5. OCR (글자 추출)
            # detail=0 은 좌표 없이 텍스트만 리스트로 반환
            ocr_result = reader.readtext(img_path, detail=0)
            ocr_text = " ".join(ocr_result)

            # 6. Captioning (장면 묘사 생성)
            raw_image = Image.open(img_path).convert('RGB')
            inputs = processor(raw_image, return_tensors="pt").to(device)
            
            # 캡션 생성 (max_new_tokens=50 정도로 길이 제한)
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)

            # 7. 데이터셋에 추가
            dataset.append({
                "scene_id": img_file,       # 파일명 (ID 역할)
                "episode": episode,         # 몇 화 (필터링용)
                "scene_no": scene_no,       # 몇 컷 (정렬용)
                "image_path": img_path,     # 파일 경로
                "scene_summary": caption,   # AI가 본 장면 설명
                "ocr_text": ocr_text        # 말풍선 내용
            })

        except Exception as e:
            print(f"⚠️ Skipping {img_file}: {e}")

    # 8. 결과 저장 (JSON)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Saved metadata to {METADATA_FILE}")
    print(f"🎉 Total {len(dataset)} items processed.")

if __name__ == "__main__":
    run_preprocessing()