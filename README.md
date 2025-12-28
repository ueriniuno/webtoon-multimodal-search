# webtoon-multimodal-search

웹툰 이미지에서 **텍스트(말풍선) 영역을 탐지**한 뒤, 해당 영역을 **CLOVA OCR로 인식**해서 텍스트를 뽑아내는 실험용 파이프라인입니다.  
(현재는 RAG/벡터DB는 아직 미포함 — “이미지 업로드 → 텍스트 잘 뽑히는지” 검증 단계)

---

## 실행 환경 세팅

### 1. 레포지토리 클론

```bash
git clone https://github.com/ueriniuno/webtoon-multimodal-search.git
cd webtoon-multimodal-search
```

---

### 2. 가상환경 생성 및 활성화

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

---

### 3. 라이브러리, 텍스트 탐지 모델 설치

```bash
pip install -r requirements.txt

pip install gdown

gdown --fuzzy "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt.onnx" -O comic_text_detector/data/comic.onnx
```

---

### 4. 환경 변수 설정
- .env 파일 예시
```env
CLOVA_OCR_INVOKE_URL=YOUR_INVOKE_URL
CLOVA_OCR_SECRET=YOUR_SECRET_KEY
```

---

### 5. 실행

```bash
cd src
python main.py
```

---

## 1) 파이프라인 개요

1. **Text Detector (comic-text-detector / ONNX)**  
   웹툰 이미지에서 말풍선/텍스트 영역 bbox 추출
2. **CLOVA OCR (API)**  
   bbox로 crop한 이미지 조각을 API로 보내 텍스트 인식 결과 수집
3. (추후) OCR + 캡셔닝 + 임베딩 → 벡터DB 저장/검색

---

## 2) 프로젝트 구조

````txt
webtoon-multimodal-search/
├─ src/
│  ├─ main.py              # 파이프라인 실행(탐지 → crop → CLOVA OCR)
│  ├─ detector.py          # 텍스트 탐지 래퍼(run_detector)
│  └─ ocr/
│     └─ clova.py          # CLOVA OCR API 클라이언트
│
├─ comic_text_detector/    # (외부 repo 기반) 텍스트 탐지 모듈
│  ├─ inference.py
│  ├─ basemodel.py
│  └─ data/
│     └─ comic.onnx        # 텍스트 탐지 모델 (git 미포함)
│
├─ images/
│  └─ sample.png           # 테스트용 이미지
│
├─ .env.example            # 환경변수 템플릿 (실제 .env는 git 미포함)
├─ requirements.txt
├─ .gitignore
└─ README.md
```



