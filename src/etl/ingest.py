# src/etl/ingest.py
import json
import uuid
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from PIL import Image
from config import QDRANT_PATH, COLLECTION_NAME, METADATA_FILE
from model_loader import models

def run_ingest():
    client = QdrantClient(path=QDRANT_PATH)
    
    # 1. DB Reset & Init
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "text": VectorParams(size=models.text_dim, distance=Distance.COSINE),
            "image": VectorParams(size=models.image_dim, distance=Distance.COSINE),
        },
    )

    # 2. Data Load
    with open(METADATA_FILE, 'r') as f:
        data = json.load(f)

    points = []
    print(f"🚀 Embedding & Indexing {len(data)} items...")

    for item in data:
        # Text Embedding (Summary + OCR)
        full_text = f"{item['scene_summary']} {item['ocr_text']}"
        text_vec = models.text_model.encode(full_text).tolist()

        # Image Embedding
        try:
            img = Image.open(item['image_path'])
            img_vec = models.image_model.encode(img).tolist()
        except:
            img_vec = [0.0] * models.image_dim

        points.append(PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, item['scene_id'])),
            vector={"text": text_vec, "image": img_vec},
            payload=item
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("✅ Ingest Complete.")

if __name__ == "__main__":
    run_ingest()