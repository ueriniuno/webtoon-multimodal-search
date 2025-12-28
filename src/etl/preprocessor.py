# src/etl/preprocessor.py
import os
import json
import torch
import easyocr
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from config import IMAGE_DIR, METADATA_FILE

def run_preprocessing():
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ '{IMAGE_DIR}' 폴더가 없습니다.")
        return

    print("🚀 Starting Pre-processing (OCR + Captioning)...")
    
    # OCR & Caption 모델 로드 (ETL 때만 쓰고 메모리 해제하기 위해 여기서 로드)
    reader = easyocr.Reader(['ko', 'en'])
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png'))]
    dataset = []

    for img_file in tqdm(image_files):
        try:
            img_path = os.path.join(IMAGE_DIR, img_file)
            
            # 1. OCR
            ocr_result = reader.readtext(img_path, detail=0)
            ocr_text = " ".join(ocr_result)

            # 2. Captioning
            raw_image = Image.open(img_path).convert('RGB')
            inputs = processor(raw_image, return_tensors="pt").to(device)
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

            dataset.append({
                "scene_id": img_file,
                "image_path": img_path,
                "scene_summary": caption,
                "ocr_text": ocr_text
            })
        except Exception as e:
            print(f"Skipping {img_file}: {e}")

    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    print(f"✅ Saved metadata to {METADATA_FILE}")

if __name__ == "__main__":
    run_preprocessing()