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
import pandas as pd

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

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
  html, body {
    background-color: #0e1117;
  }
  header[data-testid="stHeader"] {
    display: none;
  }
  .stApp { background: #0e1117; color: #ffffff; }
  .block-container { padding-top: 1.25rem; }
  
  /* 페이지 하단 배경색 통일 */
  footer {
    background-color: #0e1117 !important;
    display: none;
  }
  footer:after {
    background-color: #0e1117 !important;
  }
  .main {
    background-color: #0e1117 !important;
  }
  
  /* 채팅 입력창 하단 영역 다크 모드 */
  .stChatInputContainer {
    background-color: #0e1117 !important;
  }
  [data-testid="stBottom"] {
    background-color: #0e1117 !important;
  }
  [data-testid="stBottomBlockContainer"] {
    background-color: #0e1117 !important;
  }
  
  /* 실제 입력 필드만 정확하게 타겟팅 (border 제거) */
  [data-testid="stChatInput"] {
    background-color: transparent !important;
  }
  
  [data-testid="stChatInput"] > div {
    background-color: #1a1c24 !important;
    border-radius: 8px !important;
    border: none !important;
  }
  
  [data-testid="stChatInput"] input,
  [data-testid="stChatInput"] textarea {
    background-color: #1a1c24 !important;
    color: #e6e6e6 !important;
    border: none !important;
  }
  
  [data-testid="stChatInput"] input:focus,
  [data-testid="stChatInput"] textarea:focus {
    background-color: #1a1c24 !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
  }
  
  /* 입력창 placeholder 색상 */
  [data-testid="stChatInput"] input::placeholder,
  [data-testid="stChatInput"] textarea::placeholder {
    color: rgba(230, 230, 230, 0.5) !important;
  }
  
  /* 전송 버튼(화살표) 스타일 */
  [data-testid="stChatInput"] button {
    background-color: #0e1117 !important;
    color: rgba(230, 230, 230, 0.7) !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
  }
  
  [data-testid="stChatInput"] button:hover {
    background-color: #1a1c24 !important;
    color: rgba(230, 230, 230, 0.9) !important;
  }
  
  [data-testid="stChatInput"] button svg {
    color: rgba(230, 230, 230, 0.7) !important;
    fill: rgba(230, 230, 230, 0.7) !important;
  }
  
  [data-testid="stChatInput"] button:hover svg {
    color: rgba(230, 230, 230, 0.9) !important;
    fill: rgba(230, 230, 230, 0.9) !important;
  }
  
  /* Streamlit footer 숨기기 */
  footer[data-testid="stFooter"] {
    display: none !important;
  }
  
  /* ✅ Chat-like cards - 모든 하위 요소 강제 흰색 */
  .rag-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 14px 14px 10px 14px;
  }
  
  .rag-card,
  .rag-card *,
  .rag-card p,
  .rag-card div,
  .rag-card span {
    color: #ffffff !important;
  }
  
  .rag-title {
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 0.25rem;
    color: #ffffff !important;
  }
  .rag-meta {
    opacity: 0.85;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
    color: #ffffff !important;
  }
  
  /* 모든 서브헤더를 완전 흰색으로 변경 */
  h3, .stSubheader, [data-testid="stSubheader"] {
    color: #ffffff !important;
  }
  
  /* Subheader와 아이콘 정렬 - assistant 메시지 내부 */
  .stChatMessage[data-testid="chat-message-assistant"] h3 {
    display: inline-block !important;
    vertical-align: middle !important;
    margin: 0 !important;
    line-height: 1.5 !important;
    color: #ffffff !important;
  }
  /* ✅ write_stream / markdown 스트리밍 텍스트 강제 흰색 */
  .stChatMessage[data-testid="chat-message-assistant"]
  [data-testid="stMarkdownContainer"] *,
  .stChatMessage[data-testid="chat-message-assistant"]
  [data-testid="stMarkdownContainer"] p,
  .stChatMessage[data-testid="chat-message-assistant"]
  [data-testid="stMarkdownContainer"] span {
    color: #ffffff !important;
  }


  .stChatMessage[data-testid="chat-message-assistant"] > div {
    display: flex !important;
    align-items: flex-start !important;
  }
  
  /* ✅ Assistant 메시지 내 모든 요소 강제 흰색 */
  .stChatMessage[data-testid="chat-message-assistant"],
  .stChatMessage[data-testid="chat-message-assistant"] *,
  .stChatMessage[data-testid="chat-message-assistant"] p,
  .stChatMessage[data-testid="chat-message-assistant"] div,
  .stChatMessage[data-testid="chat-message-assistant"] span,
  .stChatMessage[data-testid="chat-message-assistant"] li {
    color: #ffffff !important;
  }
  
  /* ✅ st.caption 텍스트도 흰색 */
  .stCaptionContainer,
  .stCaptionContainer *,
  [data-testid="stCaptionContainer"],
  [data-testid="stCaptionContainer"] *,
  .caption,
  .caption * {
    color: rgba(255, 255, 255, 0.9) !important;
  }
  
  /* Make sidebar fit dark mode better */
  section[data-testid="stSidebar"] {
    background: #0b0f14;
    border-right: 1px solid rgba(255,255,255,0.06);
  }
  
  /* 사이드바 서브헤더도 흰색 */
  section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
  }
  
  /* Code block background color */
  .stCodeBlock {
    background-color: #1a1c24 !important;
  }
  pre {
    background-color: #1a1c24 !important;
    color: #e6e6e6 !important;
  }
  code {
    background-color: #1a1c24 !important;
    color: #e6e6e6 !important;
  }
  
  /* ✅ YAML 키만 #00dc64 */
  pre code .na {
    color: #00dc64 !important;
  }
  
  /* ✅ YAML 값은 흰색 */
  pre code .s, 
  pre code .s1, 
  pre code .s2,
  pre code .m,
  pre code .mi,
  pre code .mf {
    color: #ffffff !important;
  }
  
  /* Final Answer text color */
  .final-answer-text {
    color: #ffffff !important;
    background-color: #1a1c24 !important;
    padding: 1rem !important;
    border-radius: 0.5rem !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    white-space: pre-wrap !important;
    font-family: inherit !important;
    line-height: 1.6 !important;
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
    - webtoon-multimodal-search/im 폴더도 탐색
    """
    if not image_file:
        return None

    p = Path(image_file)
    if p.is_absolute() and p.exists():
        return str(p)

    data_dir = Path(settings.paths["data_dir"]).resolve()
    
    # 현재 스크립트 파일의 위치를 기준으로도 탐색
    script_dir = Path(__file__).parent.resolve()

    candidates: List[Path] = []
    
    # 1. data_dir 기준 경로들
    candidates.append(data_dir / image_file)
    candidates.append(data_dir / "images" / image_file)
    candidates.append(data_dir / "imgs" / image_file)
    candidates.append(data_dir / "im" / image_file)
    candidates.append(data_dir / "thumbnails" / image_file)
    candidates.append(data_dir / "scenes" / image_file)
    
    # 2. webtoon-multimodal-search/im 폴더 탐색 (data_dir 기준)
    current = data_dir
    for _ in range(5):  # 최대 5단계 상위 폴더까지 탐색
        webtoon_im_dir = current / "webtoon-multimodal-search" / "im"
        if webtoon_im_dir.exists():
            candidates.append(webtoon_im_dir / image_file)
            # 확장자가 없는 경우를 대비
            if p.suffix == "":
                for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                    candidates.append(webtoon_im_dir / f"{image_file}{ext}")
        current = current.parent
        if current == current.parent:  # 루트에 도달
            break
    
    # 3. 스크립트 위치 기준으로 webtoon-multimodal-search/im 탐색
    current = script_dir
    for _ in range(5):
        webtoon_im_dir = current / "webtoon-multimodal-search" / "im"
        if webtoon_im_dir.exists():
            candidates.append(webtoon_im_dir / image_file)
            if p.suffix == "":
                for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                    candidates.append(webtoon_im_dir / f"{image_file}{ext}")
        # 상위 디렉토리로 이동
        parent_webtoon_im = current.parent / "webtoon-multimodal-search" / "im"
        if parent_webtoon_im.exists():
            candidates.append(parent_webtoon_im / image_file)
            if p.suffix == "":
                for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                    candidates.append(parent_webtoon_im / f"{image_file}{ext}")
        current = current.parent
        if current == current.parent:
            break

    # 4. 확장자가 없는 경우를 대비
    if p.suffix == "":
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            candidates.append(data_dir / f"{image_file}{ext}")
            candidates.append(data_dir / "images" / f"{image_file}{ext}")
            candidates.append(data_dir / "im" / f"{image_file}{ext}")
            candidates.append(data_dir / "thumbnails" / f"{image_file}{ext}")

    # 5. 실제 존재하는 파일 찾기
    for c in candidates:
        if c.exists() and c.is_file():
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
    st.markdown('<h1 style="font-size: 3rem; margin-bottom: 0.5rem;"><span style="color: #00dc64;">ToonPT</span> 파이프라인 시각화</h1>', unsafe_allow_html=True)    
    st.caption("Input → Router → Rewriter → Reranking(Top 5) → Final Answer")

    with st.sidebar:
        st.subheader("설정")
        st.code(
            f"data_dir: {settings.paths['data_dir']}\n"
            f"qdrant_storage: {settings.paths['qdrant_storage']}\n"
            f"collection: {settings.rag['collection_name']}\n"
            f"top_k_retrieve: {settings.rag['top_k_retrieve']}\n"
            f"top_k_final: {settings.rag['top_k_final']}",
            language="javascript",
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

        # --- Reranking Similarity Score Visualization ---
        top_docs = trace.get("top_docs") or []
        if top_docs:
            st.subheader("Reranking Similarity Score (Top 5)")
            # Top 5 similarity 스코어를 그래프로 시각화
            scores_data = {}
            for i, doc in enumerate(top_docs[:5], 1):
                p = doc.payload
                label = f"{p.chapter_id}화-{p.scene_idx}컷"
                scores_data[label] = float(doc.score)
            
            # DataFrame으로 변환하여 그래프 생성
            df_scores = pd.DataFrame({
                'Document': list(scores_data.keys()),
                'Similarity Score': list(scores_data.values())
            })
            
            if PLOTLY_AVAILABLE:
                # Plotly로 꺾은선 그래프 생성 (다크 배경 + 주황색 선)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_scores['Document'],
                    y=df_scores['Similarity Score'],
                    mode='lines+markers',
                    name='Similarity Score',
                    line=dict(color='#00dc64', width=3),
                    marker=dict(color='#00dc64', size=8)
                ))
                
                # 다크 배경 설정
                fig.update_layout(
                    height=300,
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(color='#e6e6e6'),
                    xaxis=dict(
                        gridcolor='rgba(255,255,255,0.1)',
                        title=dict(text='Document', font=dict(color='#e6e6e6'))
                    ),
                    yaxis=dict(
                        gridcolor='rgba(255,255,255,0.1)',
                        title=dict(text='Similarity Score', font=dict(color='#e6e6e6'))
                    ),
                    margin=dict(l=50, r=20, t=20, b=50)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ plotly가 설치되지 않았습니다. 그래프를 표시하려면 다음 명령어로 설치하세요: `pip install plotly`")
                # 대체로 기본 차트 사용
                df_scores_indexed = df_scores.set_index('Document')
                st.line_chart(df_scores_indexed, height=300)
            
            st.caption(f"Top 5 문서의 Reranker Similarity Score 분포")

        # --- Reranking (Top 5) ---
        st.subheader("Reranking (Top 5)")
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
                        st.markdown(f'<p style="color: #00dc64; font-size: 0.85rem; margin-top: 0.25rem;">{p.image_file}</p>', unsafe_allow_html=True)
                    else:
                        # 디버깅: 어떤 image_file 값이 들어왔는지 확인
                        debug_info = f"image_file: `{p.image_file}` | data_dir: `{settings.paths['data_dir']}`"
                        st.markdown(f'<p style="color: #00dc64; font-size: 0.85rem;">썸네일을 찾지 못함: `{p.image_file}`</p>', unsafe_allow_html=True)
                        st.caption(f"디버그: {debug_info}")
                    st.caption(p.text[:120] + ("…" if len(p.text) > 120 else ""))
                    st.markdown("</div>", unsafe_allow_html=True)
                    
        # --- Final Answer ---
        st.subheader("Final Answer")
        answer = trace.get("final_answer", "")
        
        # 답변 표시 (white 색상 적용)
        if answer:
            answer_container = st.empty()
            full_text = ""
            for chunk in _stream_text(answer):
                full_text += chunk
                answer_container.markdown(f'<div class="final-answer-text">{full_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="final-answer-text">답변 생성 실패</div>', unsafe_allow_html=True)

    # 세션에 답변 저장 (한 번만)
    st.session_state.messages.append({"role": "assistant", "content": trace.get("final_answer", "")})


if __name__ == "__main__":
    main()