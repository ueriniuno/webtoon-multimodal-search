import sys
from pathlib import Path

# 프로젝트 루트 기준으로 comic-text-detector 경로 추가
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "comic_text_detector"))

from inference import inference


MODEL_PATH = str(
    ROOT / "comic_text_detector" / "data" / "comic.onnx"
)

def run_detector(image_path):
    """
    comic-text-detector 실행
    return: List[TextBlock]
    """
    text_blocks = inference(
        img_path=image_path,
        model_path=MODEL_PATH,
        device="cpu"
    )
    return text_blocks
