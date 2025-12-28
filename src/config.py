# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# ================= MODEL ZOO (실험실) =================
# 1. 임베딩 모델 (SBERT + CLIP)
TEXT_EMBEDDING_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
IMAGE_EMBEDDING_MODEL = "clip-ViT-L-14"

# 2. 리랭킹 모델 (Reranker) - 정확도 향상용
# (예: BAAI/bge-reranker-base, Dongjin-kr/ko-reranker 등)
USE_RERANKER = True
RERANKER_MODEL = "Dongjin-kr/ko-reranker"

# 3. LLM (Rewriting & Generation)
# "openai" 또는 "local"(구현 필요) 선택
LLM_TYPE = "openai" 
OPENAI_MODEL = "gpt-4o-mini"

# ================= SYSTEM SETTINGS =================
DATA_DIR = "./data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
METADATA_FILE = os.path.join(DATA_DIR, "dataset.json")

QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "webtoon_rag_v1"
TOP_K_RETRIEVAL = 5  # 1차 검색 개수
TOP_K_RERANK = 3     # 리랭킹 후 최종 개수