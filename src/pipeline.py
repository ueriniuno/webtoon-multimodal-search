# src/pipeline.py
from modules.rewriter import rewrite_query
from modules.retriever import retrieve
from modules.reranker import rerank
from modules.generator import generate_answer

def run_pipeline(user_query: str):
    print(f"\n💬 User Query: {user_query}")

    # 1. Rewriting
    refined_query = rewrite_query(user_query)
    print(f"🔄 Rewritten: {refined_query}")

    # 2. Retrieval
    candidates = retrieve(refined_query)
    print(f"🔍 Retrieved {len(candidates)} candidates.")

    # 3. Reranking
    final_docs = rerank(refined_query, candidates)
    print(f"📊 Reranked top {len(final_docs)} results.")

    # 4. Generation
    answer = generate_answer(user_query, final_docs)
    
    print("\n" + "="*30 + " FINAL ANSWER " + "="*30)
    print(answer)
    print("="*74)

if __name__ == "__main__":
    while True:
        q = input("\n질문을 입력하세요 (exit 종료): ")
        if q.lower() == "exit": break
        run_pipeline(q)