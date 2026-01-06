class RAGPipeline:
    def __init__(self, db, embedding, llm):
        """
        db: src.database.WebtoonDB 객체
        embedding: src.embedding.EmbeddingEngine 객체
        llm: src.models.QwenLLM 객체
        """
        self.db = db
        self.embedding = embedding
        self.llm = llm

    def rewrite_query(self, query):
        """
        사용자의 모호한 질문을 검색에 최적화된 키워드 중심 문장으로 확장
        """
        print("🔍 [1. Rewriter] 쿼리 확장 중...")
        prompt = (
            f"당신은 검색 전문가입니다. 다음 질문을 바탕으로 웹툰 장면 검색에 필요한 "
            f"핵심 키워드(인물 묘사, 의상, 소품, 대사)가 포함된 상세한 검색 쿼리를 작성하세요.\n"
            f"질문: {query}\n"
            f"결과:"
        )
        return self.llm.ask(prompt)

    def retrieve(self, query, top_k=3):
        """
        벡터 DB에서 유사도가 높은 문서들을 검색
        """
        print("📡 [2. Retriever] DB 검색 중...")
        vector = self.embedding.get_embeddings(query).tolist()
        # Qdrant에서 검색된 포인트 리스트 반환
        return self.db.client.query_points(
            collection_name=self.db.collection_name, 
            query=vector, 
            limit=top_k
        ).points

    def rerank(self, query, documents):
        """
        검색된 문서들 중 질문과 가장 일치하는 최상위 정보를 선택
        """
        print("⚖️ [3. Reranker] 결과 정제 중...")
        # 현재는 Qdrant에서 계산된 점수 기반 최상위 1개 추출
        # 추후 Cross-Encoder 모델을 도입하여 더 정교하게 수정 가능
        return documents[0]

    def generate_answer(self, query, context):
        print("✍️ [4. Generator] 답변 생성 중...")
        prompt = (
            f"당신은 웹툰 분석 전문가입니다. 제공된 [정보]를 바탕으로 [질문]에 대해 **필요한 답변만** 간결하게 하세요.\n\n"
            f"지침:\n"
            f"1. 질문에서 묻는 핵심 내용에 대해서만 답변하세요.\n"
            f"2. 질문과 관계없는 배경 설명이나 추가 묘사는 생략하세요.\n"
            f"3. 정보에 없는 내용은 절대 언급하지 마세요.\n"
            f"4. 한국어로만 답변하고 한자를 섞지 마세요.\n\n"
            f"[정보]: {context}\n"
            f"[질문]: {query}\n\n"
            f"결과:"
        )
        return self.llm.ask(prompt)