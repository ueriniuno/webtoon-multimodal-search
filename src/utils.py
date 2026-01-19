#src/utils.py
import json
import os
# ★ Kiwi 형태소 분석기 임포트
from kiwipiepy import Kiwi

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_text(text):
    if not text:
        return ""
    return text.replace("\n", " ").strip()

# ★ [신규 추가] 한국어 전용 토크나이저
class KoreanTokenizer:
    def __init__(self):
        # 모델 로딩 (처음에 한 번만 실행됨)
        print("🥝 Kiwi 형태소 분석기 로딩 중...")
        self.kiwi = Kiwi()
    
    def tokenize(self, text):
        """
        입력: "동구가 밥을 먹었다"
        출력: ['동구', '가', '밥', '을', '먹', '었', '다']
        """
        if not text: return []
        # 형태소 분석 후, 형태소(form)만 리스트로 반환
        return [token.form for token in self.kiwi.tokenize(text)]