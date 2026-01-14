import json
from pathlib import Path


def merge_caption_and_ocr(
    caption_dir: Path,
    ocr_dir: Path,
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1️⃣ OCR을 image_file 기준으로 map
    ocr_map = {}

    for ocr_file in ocr_dir.glob("*.json"):
        with open(ocr_file, encoding="utf-8") as f:
            data = json.load(f)
            ocr_map[data["image_file"]] = data.get("ocr", [])

    print(f"✓ OCR 로드 완료: {len(ocr_map)}개")

    # 2️⃣ caption 기준으로 merge (🔥 핵심 변경)
    merged_count = 0

    for cap_file in caption_dir.glob("*.json"):
        with open(cap_file, encoding="utf-8") as f:
            cap_data = json.load(f)

        image_file = cap_data["image_file"]
        caption = cap_data["caption"]

        ocr_blocks = ocr_map.get(image_file, [])  # 없으면 빈 리스트

        merged = {
            "image_file": image_file,
            "caption": caption,
            "ocr": ocr_blocks
        }

        out_path = output_dir / cap_file.name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        merged_count += 1

    print(f"\n✅ 병합 완료: {merged_count}개 생성")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent

    merge_caption_and_ocr(
        caption_dir=BASE_DIR / "json_data_translated",
        ocr_dir=BASE_DIR / "json_data_ocr_",
        output_dir=BASE_DIR / "json_data_merged"
    )
