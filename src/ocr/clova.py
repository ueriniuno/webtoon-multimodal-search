import requests
import uuid
import json
import cv2


class ClovaOCR:
    def __init__(self, invoke_url, secret_key):
        self.invoke_url = invoke_url
        self.secret_key = secret_key

    def run(self, image):
        # 임시 파일 저장
        cv2.imwrite("tmp_crop.jpg", image)

        request_json = {
            "images": [{"format": "jpg", "name": "crop"}],
            "requestId": str(uuid.uuid4()),
            "version": "V2",
            "timestamp": 0
        }

        payload = {
            "message": json.dumps(request_json).encode("utf-8")
        }

        files = {
            "file": open("tmp_crop.jpg", "rb")
        }

        headers = {
            "X-OCR-SECRET": self.secret_key
        }

        response = requests.post(
            self.invoke_url,
            headers=headers,
            data=payload,
            files=files
        )

        result = response.json()

        texts = []
        if result.get("images"):
            fields = result["images"][0].get("fields", [])
            for field in fields:
                texts.append({
                    "text": field.get("inferText", ""),
                    "confidence": field.get("inferConfidence", 0.0)
                })


        return texts
