import os
from sentence_transformers import SentenceTransformer
from src.ingest import setup_db_and_ingest
from src.rag_modules import QwenEngine, Rewriter, Retriever, Generator

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "qdrant_storage")
    data_folder = os.path.join(current_dir, "data")

    if not os.path.exists(data_folder): os.makedirs(data_folder)

    print("🚀 임베딩 모델 로드 중...")
    embed_model = SentenceTransformer('BAAI/bge-m3')

    # 모든 JSON 통합 적재
    client, col_name = setup_db_and_ingest(db_path, embed_model, data_folder)

    engine = QwenEngine()
    rewriter, retriever, generator = Rewriter(engine), Retriever(client, col_name, embed_model), Generator(engine)

    user_query = "주인공이 입고 있는 옷과 주변 사물들에 대해 자세히 알려줘."
    refined_q = rewriter.rewrite(user_query)
    doc = retriever.search(refined_q)
    
    if doc:
        ans = generator.answer(doc.payload['full_text'], user_query)
        print("\n" + "="*50 + f"\n🎯 답변:\n{ans}\n" + "="*50)

if __name__ == "__main__":
    main()