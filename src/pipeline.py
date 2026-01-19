# src/pipeline.py
import json
import os
from rank_bm25 import BM25Okapi
from qdrant_client.http import models

from src.config import settings
from src.schemas import SearchResult, ScenePayload
from src.router import LLMRouter
from src.expander import QueryExpander
from src.reranker import Reranker
from src.prompts import RAG_GENERATION, RAG_SYSTEM 
# 👇 [수정] KoreanTokenizer 추가
from src.utils import load_json, KoreanTokenizer

class RAGPipeline:
    def __init__(self, db, embedding, llm):
        self.db = db
        self.embedding = embedding
        self.llm = llm
        
        print("🔧 [Pipeline] 컴포넌트 로딩...")
        
        self.lookup = load_json(settings.paths["lookup_store"])
        if not self.lookup:
            print("⚠️ [Warning] Lookup Store 비어있음.")
        
        # 👇 [수정] 토크나이저 초기화 (Kiwi)
        self.tokenizer = KoreanTokenizer()

        # BM25 엔진 초기화
        self.bm25 = None
        self.bm25_data = [] 
        self._init_bm25()
        
        self.router = LLMRouter(llm)
        self.expander = QueryExpander(llm)
        self.reranker = Reranker()

    def _init_bm25(self):
        bm25_path = os.path.join(settings.paths["data_dir"], "bm25_corpus.json")
        if not os.path.exists(bm25_path):
            print("⚠️ [Warning] BM25 코퍼스 파일이 없습니다.")
            return
            
        print("   📘 BM25 인덱스 생성 중 (형태소 분석 적용)...")
        with open(bm25_path, 'r', encoding='utf-8') as f:
            self.bm25_data = json.load(f)
        
        # 👇 [수정] split() 대신 형태소 분석기 사용
        tokenized_corpus = [self.tokenizer.tokenize(doc['text']) for doc in self.bm25_data]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("   ✅ BM25 준비 완료.")

    def hybrid_search(self, query, expanded_query, top_k=50):
        """
        [Vector Search] -> Expanded Query 사용 (의미 파악)
        [BM25 Search] -> Original + Expanded 혼합 사용 (안전성 + 확장성)
        """
        # 1. Vector Search
        query_vector = self.embedding.get_embeddings(expanded_query).tolist()
        vector_hits = self.db.search(query_vector=query_vector, limit=top_k)
        
        # 2. BM25 Search
        bm25_hits = []
        if self.bm25:
            # 원본과 확장된 쿼리를 합쳐서 BM25에 넣습니다.
            combined_query = f"{query} {expanded_query}" 
            
            # 👇 [수정] 검색어도 형태소 분석기로 쪼갬 ("동구가" -> ['동구', '가'])
            tokenized_query = self.tokenizer.tokenize(combined_query)
            
            doc_scores = self.bm25.get_scores(tokenized_query)
            top_indexes = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
            
            for idx in top_indexes:
                hit_data = self.bm25_data[idx]
                bm25_hits.append(models.ScoredPoint(
                    id=hit_data['id'],
                    version=0,
                    score=doc_scores[idx],
                    payload=hit_data['payload'],
                    vector=None
                ))

        # 3. RRF (Reciprocal Rank Fusion)
        k = 60
        fused_scores = {}
        
        for rank, hit in enumerate(vector_hits):
            if hit.id not in fused_scores:
                fused_scores[hit.id] = {"score": 0, "payload": hit.payload, "obj": hit}
            fused_scores[hit.id]["score"] += 1 / (k + rank + 1)
            
        for rank, hit in enumerate(bm25_hits):
            if hit.id not in fused_scores:
                fused_scores[hit.id] = {"score": 0, "payload": hit.payload, "obj": hit}
            fused_scores[hit.id]["score"] += 1 / (k + rank + 1)
            
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x]["score"], reverse=True)
        
        final_results = []
        for pid in sorted_ids[:top_k]:
            item = fused_scores[pid]
            original_pt = item['obj']
            final_results.append(models.ScoredPoint(
                id=original_pt.id,
                version=0,
                score=item['score'],
                payload=item['payload'],
                vector=None
            ))
            
        return final_results

    def _fetch_window_context(self, points, window_size=0):
        if window_size <= 0:
            return {p.id: p.payload['text'] for p in points}

        ids_to_fetch = set()
        for p in points:
            center_id = p.id
            ids_to_fetch.add(center_id)
            for i in range(1, window_size + 1):
                ids_to_fetch.add(center_id - i)
                ids_to_fetch.add(center_id + i)

        fetched_points = self.db.client.retrieve(
            collection_name=settings.rag["collection_name"],
            ids=list(ids_to_fetch)
        )
        text_map = {fp.id: fp.payload['text'] for fp in fetched_points}

        merged_texts = {}
        for p in points:
            center_id = p.id
            full_text_list = []
            for i in range(window_size, 0, -1):
                t = text_map.get(center_id - i)
                if t: full_text_list.append(t)
            
            full_text_list.append(text_map.get(center_id, ""))
            
            for i in range(1, window_size + 1):
                t = text_map.get(center_id + i)
                if t: full_text_list.append(t)
            
            merged_texts[center_id] = " ".join(full_text_list)
            
        return merged_texts

    def _save_debug_log(self, query, expanded_query, scanned_points, final_docs):
        log_path = "debug_search_log.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"=== Debug Log (Hybrid Search) ===\n")
            f.write(f"Original Query: {query}\n")
            f.write(f"Expanded Query: {expanded_query}\n\n")
            
            f.write(f"=== 1. Hybrid Retrieval Candidates (RRF Top {len(scanned_points)}) ===\n")
            for i, p in enumerate(scanned_points):
                f.write(f"[{i+1}] ID: {p.id} | RRF Score: {p.score:.6f}\n")
                f.write(f"    Text: {p.payload['text'][:100]}...\n")
            
            f.write(f"\n=== 2. Reranker Selected (Top {len(final_docs)}) ===\n")
            for i, doc in enumerate(final_docs):
                p = doc.payload
                f.write(f"[{i+1}] ID: {p.id} | Rerank Score: {doc.score:.4f}\n")
                f.write(f"    Full Context used:\n{doc.full_context_text[:200]}...\n\n")
        
        print(f"📝 [Log] 검색 상세 내용이 '{log_path}'에 저장되었습니다.")

    def run(self, query, window_size=0):
        print(f"\n🚀 [Pipeline] 처리 시작: '{query}' (Hybrid Mode, Window: {window_size})")
        
        intent, cid = self.router.route(query)
        print(f"🚦 [Router] 분석 결과: Intent='{intent}', Chapter='{cid}'")
        
        if intent == "lookup_chapter" and cid:
            summary = self.lookup.get(f"chapter_{cid}", "정보 없음")
            return self.llm.ask("당신은 요약 봇입니다.", f"{cid}화 내용을 요약해줘.\n내용: {summary}")

        search_query = self.expander.expand(query)
        print(f"🔍 [Expander] 확장된 쿼리: '{search_query}'")

        scanned_points = self.hybrid_search(query, search_query, top_k=settings.rag["top_k_retrieve"])
        
        if not scanned_points: return "검색 결과가 없습니다."

        window_texts = self._fetch_window_context(scanned_points, window_size=window_size)

        candidates = []
        for hit in scanned_points:
            p = hit.payload
            center_id = hit.id
            
            extended_text = window_texts.get(center_id, p['text'])
            c_txt = self.lookup.get(f"chapter_{p['chapter_id']}", "")
            e_txt = self.lookup.get(f"event_{p.get('event_id')}", "")
            
            full_context_for_rerank = (
                f"{extended_text}\n\n"
                f"[참고 - 사건: {e_txt}]\n"
                f"[참고 - 전체: {c_txt}]"
            )
            candidates.append(SearchResult(payload=ScenePayload(**p), full_context_text=full_context_for_rerank))

        final_docs = self.reranker.rerank(query=query, docs=candidates, top_k=settings.rag["top_k_final"])
        self._save_debug_log(query, search_query, scanned_points, final_docs)
        print(f"🎯 [Reranker] {len(scanned_points)}개 -> {len(final_docs)}개 선정")

        events = set()
        chapters = set()
        scenes = []
        
        for doc in final_docs:
            p = doc.payload
            scenes.append(f"- [{p.chapter_id}화 {p.scene_idx}컷] {p.text}")
            
            c_full = self.lookup.get(f"chapter_{p.chapter_id}", "")
            if c_full: chapters.add(f"- [Ch.{p.chapter_id}] {c_full}")
            if p.event_id:
                e_full = self.lookup.get(f"event_{p.event_id}", "")
                if e_full: events.add(f"- [Event] {e_full}")

        final_prompt = RAG_GENERATION.format(
            # ★ [수정] query -> search_query로 변경! (확장된 쿼리를 넘겨야 이름 매칭이 잘 됨)
            character_info=self.expander.get_profile_str(search_query),
            global_summary=self.lookup.get("global", ""),
            context_summaries="\n".join(events) + "\n" + "\n".join(chapters),
            scene_details="\n".join(scenes),
            user_query=query
        )

        return self.llm.ask(RAG_SYSTEM, final_prompt)