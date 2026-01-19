#src/database.py
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.config import settings

class WebtoonDB:
    def __init__(self):
        # config.yaml에 지정된 경로로 DB 연결
        self.client = QdrantClient(path=settings.paths["qdrant_storage"])
        self.collection_name = settings.rag["collection_name"]

    def ensure_collection(self, vector_size=768):
        """컬렉션이 없으면 새로 생성"""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, 
                    distance=models.Distance.COSINE
                )
            )
            # 필터링 검색 속도 향상을 위한 인덱스 생성
            self.client.create_payload_index(self.collection_name, "chapter_id", models.PayloadSchemaType.INTEGER)
            print(f"✅ 컬렉션 생성 완료: {self.collection_name}")

    def upsert(self, points):
        """데이터 업로드"""
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    # 👇 [수정됨] search -> query_points로 변경
    def search(self, query_vector, limit=50):
        """벡터 검색 수행"""
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector, # query_points에서는 인자 이름이 'query'입니다
            limit=limit
        )
        # query_points는 객체를 반환하므로 .points로 리스트만 꺼내줘야 합니다.
        return response.points