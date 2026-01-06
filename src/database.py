from qdrant_client import QdrantClient
from qdrant_client.http import models

class WebtoonDB:
    def __init__(self, db_path):
        self.client = QdrantClient(path=db_path)
        self.collection_name = "webtoon_db"

    def ensure_collection(self, vector_size=1024):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )

    def upsert_data(self, points):
        self.client.upsert(collection_name=self.collection_name, points=points)