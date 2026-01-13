# ocr_preprocess.py
import json
from pathlib import Path


def normalize_filename(filename: str) -> str:
    """
    OCR filename을 Joy Caption과 동일하게 맞춤
    - '-2_' 제거
    """
    return filename.replace("-2_", "_")


def flatten_blocks(blocks: list) -> list:
    """
    block 내부 texts를 한 줄 string으로 변환
    """
    flattened = []

    for block in blocks:
        texts = [
            t["text"].strip()
            for t in block.get("texts", [])
            if t.get("text")
        ]

        if not texts:
            continue

        flattened.append({
            "block_number": block["block_number"],
            "text": " ".join(texts)
        })

    return flattened


def split_ocr_json(
    input_path: Path,
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    for item in ocr_data:
        image_file = normalize_filename(item["filename"])
        blocks = flatten_blocks(item.get("blocks", []))

        if not blocks:
            continue

        result = {
            "image_file": image_file,
            "ocr": blocks
        }

        out_path = output_dir / f"{Path(image_file).stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
