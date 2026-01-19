#src/run_ingest.py
import json
import glob
import os
import re
from tqdm import tqdm
from qdrant_client.http import models

from src.config import settings
from src.database import WebtoonDB
from src.embedding import EmbeddingEngine
from src.schemas import ScenePayload

# =========================================================
# 🛠️ 유틸리티 함수
# =========================================================

def flatten_scene_text(text_data):
    """ 리스트+딕셔너리 텍스트 평탄화 """
    if isinstance(text_data, str): 
        return text_data
    
    combined = []
    if isinstance(text_data, list):
        for item in text_data:
            if isinstance(item, str):
                combined.append(item)
            elif isinstance(item, dict):
                ocr = item.get("ocr", "")
                if isinstance(ocr, list): 
                    ocr = " ".join(ocr)
                if str(ocr).strip(): 
                    combined.append(f"ocr: {str(ocr)}")
    
    return " ".join(combined)

def parse_episode_range(range_str):
    try:
        match = re.search(r'(\d+)\s*~\s*(\d+)', str(range_str))
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            return list(range(start, end + 1))
        match = re.search(r'(\d+)', str(range_str))
        if match:
            return [int(match.group(1))]
    except:
        return []
    return []

# =========================================================
# 🚀 메인 적재 로직
# =========================================================

def run_ingest():
    print(f"\n🏗️  [Data Ingestion] 데이터 적재 시작 (Hybrid Mode)")
    
    # -----------------------------------------------------
    # 1. Lookup Store 구축
    # -----------------------------------------------------
    lookup_store = {}
    chapter_to_event = {}

    # Global
    g_path = os.path.join(settings.paths["data_dir"], "global_summary.json")
    if os.path.exists(g_path):
        with open(g_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            lookup_store["global"] = data.get("full_synopsis", "")

    # Events
    event_files = glob.glob(os.path.join(settings.paths["data_dir"], "events/*.json"))
    for f in event_files:
        with open(f, 'r', encoding='utf-8') as file:
            data = json.load(file)
            eid = os.path.basename(f).replace(".json", "")
            lookup_store[f"event_{eid}"] = data.get('content', "")
            
            chapters = parse_episode_range(data.get("episode_range", ""))
            for cid in chapters:
                chapter_to_event[cid] = eid

    # Chapters
    chapter_files = glob.glob(os.path.join(settings.paths["data_dir"], "chapter_summaries/*.json"))
    for f in chapter_files:
        with open(f, 'r', encoding='utf-8') as file:
            data = json.load(file)
            ep_str = data.get("episode", "0")
            cid_match = re.search(r'\d+', str(ep_str))
            if not cid_match: continue
            
            cid = int(cid_match.group())
            text = data.get("summary", "")
            lookup_store[f"chapter_{cid}"] = text
            
    with open(settings.paths["lookup_store"], "w", encoding='utf-8') as f:
        json.dump(lookup_store, f, ensure_ascii=False, indent=2)
    print("   💾 Lookup Store 저장 완료.")

    # -----------------------------------------------------
    # 2. Vector DB & BM25 Corpus 적재
    # -----------------------------------------------------
    print("\n🚀 [Vector DB] 씬 데이터 처리 시작...")
    
    db = WebtoonDB()
    db.ensure_collection()
    embedder = EmbeddingEngine()

    scene_files = glob.glob(os.path.join(settings.paths["data_dir"], "scenes/*.json"))
    
    batch_points = []
    BATCH_SIZE = 50
    total_count = 0
    
    # ★ BM25용 데이터 저장소
    bm25_corpus = []

    for f in tqdm(scene_files, desc="Embedding & indexing"):
        with open(f, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            fname = data.get("image_file", "")
            match = re.search(r'(\d+)_(\d+)', fname)
            if not match: continue
            cid, sidx = int(match.group(1)), int(match.group(2))
            
            raw_text = flatten_scene_text(data.get("full_text", []))
            if not raw_text.strip(): continue

            # 1. Vector DB용 임베딩
            embedding_input = raw_text 
            vector = embedder.get_embeddings(embedding_input).tolist()
            
            unique_id = cid * 10000 + sidx
            payload = ScenePayload(
                id=f"{cid}_{sidx}",
                text=raw_text,
                chapter_id=cid,
                scene_idx=sidx,
                event_id=chapter_to_event.get(cid),
                image_file=fname
            )
            
            batch_points.append(models.PointStruct(
                id=unique_id,
                vector=vector,
                payload=payload.model_dump()
            ))
            
            # 2. ★ BM25용 데이터 수집 (순수 씬 텍스트만)
            bm25_corpus.append({
                "id": unique_id,
                "text": raw_text,
                "payload": payload.model_dump() # 나중에 BM25 결과만으로도 복원 가능하게
            })
            
            # 배치 저장
            if len(batch_points) >= BATCH_SIZE:
                db.upsert(batch_points)
                total_count += len(batch_points)
                batch_points = []

    if batch_points:
        db.upsert(batch_points)
        total_count += len(batch_points)

    # 3. ★ BM25 코퍼스 파일 저장
    bm25_path = os.path.join(settings.paths["data_dir"], "bm25_corpus.json")
    with open(bm25_path, "w", encoding='utf-8') as f:
        json.dump(bm25_corpus, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 총 {total_count}개의 씬 적재 완료.")
    print(f"   📘 BM25 코퍼스 저장 완료: {bm25_path}")

if __name__ == "__main__":
    run_ingest()