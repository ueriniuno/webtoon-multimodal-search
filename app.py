import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Iterable

# 필수 패키지 확인 및 에러 핸들링
MISSING_PACKAGES = []

try:
    import streamlit as st
except ImportError:
    MISSING_PACKAGES.append("streamlit")

try:
    from qdrant_client import QdrantClient
except ImportError:
    MISSING_PACKAGES.append("qdrant-client")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    MISSING_PACKAGES.append("sentence-transformers")

try:
    import torch
except ImportError:
    MISSING_PACKAGES.append("torch")

if MISSING_PACKAGES:
    print("=" * 60)
    print("❌ 필수 패키지가 설치되지 않았습니다!")
    print("=" * 60)
    print(f"누락된 패키지: {', '.join(MISSING_PACKAGES)}")
    print("\n설치 방법:")
    print(f"  {sys.executable} -m pip install {' '.join(MISSING_PACKAGES)}")
    print("\n또는 설치 스크립트 실행:")
    print("  python install_dependencies.py")
    print("=" * 60)
    sys.exit(1)

import streamlit as st

# src 모듈 임포트 (에러 핸들링)
try:
    from src import WebtoonDB, EmbeddingEngine, ExaoneLLM, RAGPipeline
    from src.config import settings
    from src.prompts import (
        RAG_GENERATION_CHAPTER,
        RAG_GENERATION_SCENE,
        RAG_SYSTEM_CHAPTER,
        RAG_SYSTEM_SCENE,
    )
except ImportError as e:
    st.error(f"❌ 모듈 임포트 실패: {e}")
    st.info(
        f"""
        **필수 패키지 설치가 필요합니다:**
        
        터미널에서 다음 명령어를 실행하세요:
        ```bash
        {sys.executable} -m pip install qdrant-client sentence-transformers transformers torch PyYAML rank-bm25 kiwipiepy
        ```
        
        또는 설치 스크립트를 실행:
        ```bash
        python install_dependencies.py
        ```
        """
    )
    st.stop()


st.set_page_config(
    page_title="Webtoon RAG Pipeline Viewer",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)


_DARK_CSS = """

<style>
  header[data-testid="stHeader"] {
    display: none;
  }
  .stApp { background: #0e1117; color: #e6e6e6; }
  .block-container { padding-top: 1.25rem; }
  /* Chat-like cards */
  .rag-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 14px 14px 10px 14px;
  }
  .rag-title {
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 0.25rem;
  }
  .rag-meta {
    opacity: 0.85;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
  }
  /* Make sidebar fit dark mode better */
  section[data-testid="stSidebar"] {
    background: #0b0f14;
    border-right: 1px solid rgba(255,255,255,0.06);
  }
</style>
"""
st.markdown(_DARK_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="모델/DB 초기화 중… (최초 1회만 오래 걸려요)")
def get_pipeline() -> RAGPipeline:
    """
    config/config.yaml(settings)을 참조하여 컴포넌트를 초기화하고,
    Streamlit 재실행 시에도 리소스를 재사용합니다.
    """
    db = WebtoonDB()
    embedder = EmbeddingEngine()
    llm = ExaoneLLM()
    return RAGPipeline(db, embedder, llm)


def _resolve_image_path(image_file: str) -> Optional[str]:
    """
    검색 결과 payload의 image_file을 실제 파일 경로로 해석합니다.
    - 절대경로면 그대로 사용
    - 상대경로면 settings.paths['data_dir'] 기준으로 여러 후보를 탐색
    """
    if not image_file:
        return None

    p = Path(image_file)
    if p.is_absolute() and p.exists():
        return str(p)

    data_dir = Path(settings.paths["data_dir"])

    candidates: List[Path] = []
    # 흔한 구조 후보들
    candidates.append(data_dir / image_file)
    candidates.append(data_dir / "images" / image_file)
    candidates.append(data_dir / "imgs" / image_file)
    candidates.append(data_dir / "thumbnails" / image_file)
    candidates.append(data_dir / "scenes" / image_file)

    # 확장자가 없는 경우를 대비
    if p.suffix == "":
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            candidates.append(data_dir / f"{image_file}{ext}")
            candidates.append(data_dir / "images" / f"{image_file}{ext}")
            candidates.append(data_dir / "thumbnails" / f"{image_file}{ext}")

    for c in candidates:
        if c.exists():
            return str(c)

    return None


def _stream_text(text: str, chunk: int = 40) -> Iterable[str]:
    """LLM이 스트리밍을 지원하지 않으니, UI에서만 텍스트를 나눠서 흘려보냅니다."""
    if not text:
        return
    for i in range(0, len(text), chunk):
        yield text[i : i + chunk]


def _get_webtoon_title() -> str:
    """
    데이터셋이 단일 웹툰인 경우가 많아 기본 타이틀을 제공.
    데이터 폴더에 metadata/global_summary 등이 있으면 거기서도 시도합니다(없으면 기본값).
    """
    data_dir = Path(settings.paths["data_dir"])
    for candidate in [data_dir / "metadata.json", data_dir / "meta.json", data_dir / "global_summary.json"]:
        try:
            if candidate.exists():
                import json

                obj = json.loads(candidate.read_text(encoding="utf-8"))
                for key in ["title", "webtoon_title", "name"]:
                    v = obj.get(key)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
        except Exception:
            pass
    return "Webtoon"


def run_pipeline_with_traces(pipeline: RAGPipeline, query: str, window_size: int = 0) -> dict:
    """
    src/pipeline.py의 run() 흐름을 그대로 따라가되,
    UI에서 보여줄 중간 결과(의도/재작성/Top5 문서)를 같이 반환합니다.
    """
    intent, cid = pipeline.router.route(query)

    trace = {
        "query": query,
        "intent": intent,
        "chapter_id": cid,
        "rewritten_query": None,
        "top_docs": [],
        "final_answer": "",
        "mode": "search",
    }

    # Case A: 챕터 요약(lookup)
    if intent == "lookup_chapter" and cid:
        trace["mode"] = "lookup_chapter"

        chapter_summary = pipeline.lookup.get(f"chapter_{cid}", "정보 없음")
        event_id = pipeline.chapter_event_map.get(cid)
        event_summary = pipeline.lookup.get(f"event_{event_id}", "") if event_id else ""

        full_context = ""
        if event_summary:
            full_context += f"[Related Event Summary (Event {event_id})]\n{event_summary}\n\n"
        full_context += f"[Target Chapter Summary (Chapter {cid})]\n{chapter_summary}"

        formatted_prompt = RAG_GENERATION_CHAPTER.format(
            user_query=query,
            character_info=pipeline.raw_character_info,
            global_summary=pipeline.lookup.get("global", ""),
            context_summaries=full_context,
        )
        trace["final_answer"] = pipeline.llm.ask(RAG_SYSTEM_CHAPTER, formatted_prompt)
        return trace

    # Case B: 일반 검색(search)
    rewritten = pipeline.expander.expand(query)
    trace["rewritten_query"] = rewritten

    scanned_points = pipeline.hybrid_search(query, rewritten, top_k=settings.rag["top_k_retrieve"])
    if not scanned_points:
        trace["final_answer"] = "검색 결과가 없습니다."
        return trace

    window_texts = pipeline._fetch_window_context(scanned_points, window_size=window_size)

    candidates = []
    for hit in scanned_points:
        p = hit.payload
        center_id = hit.id

        extended_text = window_texts.get(center_id, p["text"])
        c_txt = pipeline.lookup.get(f"chapter_{p['chapter_id']}", "")
        e_txt = pipeline.lookup.get(f"event_{p.get('event_id')}", "")

        full_context_for_rerank = (
            f"{extended_text}\n\n"
            f"[참고 - 사건: {e_txt}]\n"
            f"[참고 - 전체: {c_txt}]"
        )
        from src.schemas import SearchResult, ScenePayload

        candidates.append(SearchResult(payload=ScenePayload(**p), full_context_text=full_context_for_rerank))

    final_docs = pipeline.reranker.rerank(query=query, docs=candidates, top_k=settings.rag["top_k_final"])
    trace["top_docs"] = final_docs

    events = set()
    chapters = set()
    scenes = []

    for doc in final_docs:
        p = doc.payload
        scenes.append(f"- [{p.chapter_id}화 {p.scene_idx}컷] {p.text}")

        c_full = pipeline.lookup.get(f"chapter_{p.chapter_id}", "")
        if c_full:
            chapters.add(f"- [Ch.{p.chapter_id}] {c_full}")
        if p.event_id:
            e_full = pipeline.lookup.get(f"event_{p.event_id}", "")
            if e_full:
                events.add(f"- [Event] {e_full}")

    final_prompt = RAG_GENERATION_SCENE.format(
        character_info=pipeline.raw_character_info,
        global_summary=pipeline.lookup.get("global", ""),
        context_summaries="\n".join(events) + "\n" + "\n".join(chapters),
        scene_details="\n".join(scenes),
        user_query=query,
    )
    trace["final_answer"] = pipeline.llm.ask(RAG_SYSTEM_SCENE, final_prompt)
    return trace


def main():
    st.title("RAG 파이프라인 시각화")
    st.caption("Input → Router → Rewriter → Reranking(Top 5) → Final Answer")

    with st.sidebar:
        st.subheader("설정")
        st.write("`config/config.yaml` 기준으로 초기화됩니다.")
        st.code(
            f"data_dir: {settings.paths['data_dir']}\n"
            f"qdrant_storage: {settings.paths['qdrant_storage']}\n"
            f"collection: {settings.rag['collection_name']}\n"
            f"top_k_retrieve: {settings.rag['top_k_retrieve']}\n"
            f"top_k_final: {settings.rag['top_k_final']}",
            language="yaml",
        )
        window_size = st.slider("Window size (앞뒤 문맥 확장)", 0, 3, 0, 1)
        st.divider()
        st.info("라우팅/리라이트/리랭킹은 실제 파이프라인 컴포넌트를 그대로 호출합니다.")

    pipeline = get_pipeline()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 기존 대화 렌더
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_query = st.chat_input("질문을 입력하세요")
    if not user_query:
        return

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 실행
    with st.chat_message("assistant"):
        with st.spinner("파이프라인 실행 중…"):
            trace = run_pipeline_with_traces(pipeline, user_query, window_size=window_size)

        # --- Router ---
        st.subheader("Router")
        intent = trace.get("intent")
        cid = trace.get("chapter_id")
        if intent == "lookup_chapter":
            st.info(f"Intent: **{intent}** · Chapter: **{cid}**")
        else:
            st.info(f"Intent: **{intent}**")

        # --- Rewriter ---
        st.subheader("Rewriter")
        if trace.get("rewritten_query"):
            st.code(trace["rewritten_query"], language="text")
        else:
            st.caption("lookup 모드에서는 Rewriter 단계가 생략됩니다.")

        # --- Reranking (Top 5) ---
        st.subheader("Reranking (Top 5)")
        top_docs = trace.get("top_docs") or []
        if not top_docs:
            st.caption("lookup 모드 또는 검색 결과 없음으로 인해 Top 5 문서가 없습니다.")
        else:
            webtoon_title = _get_webtoon_title()
            cols = st.columns(5, gap="small")
            for i, doc in enumerate(top_docs[:5]):
                p = doc.payload
                score = float(doc.score)
                title = webtoon_title
                subtitle = f"{p.chapter_id}화 · 씬 {p.scene_idx}"
                img_path = _resolve_image_path(p.image_file)

                with cols[i]:
                    st.markdown('<div class="rag-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="rag-title">{title}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="rag-meta">{subtitle}<br/>Similarity: <b>{score:.4f}</b></div>',
                        unsafe_allow_html=True,
                    )
                    if img_path:
                        st.image(img_path, use_container_width=True)
                    else:
                        st.caption(f"썸네일을 찾지 못함: `{p.image_file}`")
                    st.caption(p.text[:120] + ("…" if len(p.text) > 120 else ""))
                    st.markdown("</div>", unsafe_allow_html=True)

        # --- Final Answer ---
        st.subheader("Final Answer")
        answer = trace.get("final_answer", "")
        st.write_stream(_stream_text(answer)) if answer else st.write("답변 생성 실패")

    st.session_state.messages.append({"role": "assistant", "content": trace.get("final_answer", "")})


if __name__ == "__main__":
    # Streamlit 실행 시에는 이 블록이 실행되지 않지만,
    # python app.py 형태로 실행하는 경우를 대비합니다.
    main()

