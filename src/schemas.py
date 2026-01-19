# src/schemas.py
from pydantic import BaseModel
from typing import Optional, List

# 1. 벡터 DB(Qdrant)에 저장될 Payload 구조
class ScenePayload(BaseModel):
    id: str             # 고유 ID (예: "2_005")
    text: str           # 4컷 묶음 원본 텍스트 (OCR 포함)
    chapter_id: int     # 챕터 번호
    scene_idx: int      # 씬 번호
    event_id: Optional[str] = None # 사건 ID (없을 수도 있음)
    image_file: str     # 원본 파일명
    type: str = "scene_group"

# 2. 검색 파이프라인 내부에서 쓰일 검색 결과 객체
class SearchResult(BaseModel):
    payload: ScenePayload       # 위에서 정의한 씬 데이터
    score: float = 0.0          # 검색 점수 (Vector 유사도 or Rerank 점수)
    
    # Reranker에게 보여주기 위해 [사건+화+씬]을 합친 텍스트
    # (DB에는 저장 안 함, 검색 도중에만 생성)
    full_context_text: Optional[str] = None