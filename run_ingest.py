import os, json, glob
from qdrant_client.http import models
from src.database import WebtoonDB
from src.embedding import EmbeddingEngine

def run():
    db = WebtoonDB("./qdrant_storage")
    db.ensure_collection()
    embedder = EmbeddingEngine()

    json_files = glob.glob("data/*.json")
    for f_path in json_files:
        with open(f_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 모든 데이터를 텍스트화 (사용자님 제공 텍스트 포함)
            full_text = str(data) 
            vector = embedder.get_embeddings(full_text).tolist()
            
            point = models.PointStruct(
                id=hash(f_path) % 10**8, # 파일명 기반 고유 ID
                vector=vector,
                payload={"full_text": full_text, "source": f_path}
            )
            db.upsert_data([point])
            print(f"✅ {f_path} 데이터 적재 완료")

if __name__ == "__main__":
    run()