# src/pipeline.py
from qdrant_client.http import models

class RAGPipeline:
    def __init__(self, db, embedding, llm):
        self.db = db
        self.embedding = embedding
        self.llm = llm

    def retrieve(self, query, top_k_chapters=3, top_k_anchors=5, window_size=0):
        """
        [병렬 검색 로직]
        1. 챕터 검색: 줄거리 맥락(Global Context) 확보용 (씬 검색 필터링 X)
        2. 씬 검색: 전체 DB 대상 독립적 검색 (Local Context)
        3. 반환: (씬_리스트, 요약본_리스트)
        """
        query_vector = self.embedding.get_embeddings(query).tolist()

        # --- Step 1: 챕터 검색 (독립적 수행) ---
        print(f"📡 [Step 1] 챕터(Global Context) 검색 (Top-{top_k_chapters})...")
        
        chapters = self.db.client.query_points(
            collection_name=self.db.chapter_col,
            query=query_vector,
            limit=top_k_chapters
        ).points
        
        relevant_summaries = []
        
        if not chapters:
            print("   ⚠️ 관련 챕터 없음")
        else:
            print(f"   🔎 확보된 줄거리 맥락:")
            for c in chapters:
                ch_id = c.payload['chapter_id']
                score = c.score
                
                # LLM에게 줄 요약 텍스트 수집
                summary_text = f"[Chapter {ch_id} 요약] {c.payload['summary']}"
                relevant_summaries.append(summary_text)
                
                print(f"     - Ch.{ch_id} (유사도: {score:.4f})")

        # --- Step 2: 씬 검색 (전체 범위 대상 수행) ---
        # [변경점] 챕터 ID로 필터링(query_filter)을 걸지 않습니다!
        print(f"📡 [Step 2] 씬(Local Context) 전체 검색 (Top-{top_k_anchors})...")

        scenes = self.db.client.query_points(
            collection_name=self.db.scene_col,
            query=query_vector,
            query_filter=None,  # 👈 핵심: 필터 없이 전체 검색
            limit=top_k_anchors
        ).points

        # --- Step 3: 윈도우 확장 ---
        final_scene_ids = set()
        
        for anchor in scenes:
            ch_id = anchor.payload['chapter_id']
            center_idx = anchor.payload['scene_idx']
            
            # window_size 만큼 앞뒤 확장
            for i in range(center_idx - window_size, center_idx + window_size + 1):
                if i < 1: continue 
                expanded_id = ch_id * 10000 + i
                final_scene_ids.add(expanded_id)

        if not final_scene_ids:
            return [], relevant_summaries # 씬은 없어도 요약은 반환

        # --- Step 4: 최종 데이터 조회 ---
        print(f"📡 [Step 3] 최종 {len(final_scene_ids)}개 컷 데이터 로딩 (Window: +/-{window_size})")
        retrieved_points = self.db.client.retrieve(
            collection_name=self.db.scene_col,
            ids=list(final_scene_ids)
        )
        
        final_docs = sorted(retrieved_points, key=lambda x: (x.payload['chapter_id'], x.payload['scene_idx']))
        
        return final_docs, relevant_summaries

    def generate_answer(self, query, results_tuple):
        """
        results_tuple: (final_docs, summaries)
        """
        documents, summaries = results_tuple
        
        print(f"✍️ [Generation] 답변 생성 중 (요약본 {len(summaries)}개 + 씬 {len(documents)}개 참고)...")
        
        # 1. Global Context
        summary_context = "\n".join(summaries)
        
        # 2. Local Context
        scene_context_list = []
        for doc in documents:
            p = doc.payload
            scene_desc = f"[{p['chapter_id']}화 {p['scene_idx']}컷] {p['full_text']}"
            scene_context_list.append(scene_desc)
        
        full_scene_context = "\n\n".join(scene_context_list)
        
        system_msg = "당신은 웹툰 분석 전문가입니다. [전체 줄거리]와 [상세 장면]을 모두 고려하여 질문에 답변하세요."
        
        user_msg = f"""
### 1. 전체 줄거리 맥락 (Global Context):
{summary_context}

### 2. 상세 장면 맥락 (Local Context):
{full_scene_context}

### 질문:
{query}

### 답변:
"""
        return self.llm.ask(system_msg, user_msg)