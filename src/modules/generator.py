# src/modules/generator.py
from openai import OpenAI
from config import OPENAI_MODEL

client = OpenAI()

def generate_answer(user_query: str, contexts: list) -> str:
    # Context 구성
    context_str = ""
    for i, doc in enumerate(contexts):
        info = doc.payload
        context_str += f"[{i+1}] 장면묘사: {info['scene_summary']}, 대사: {info['ocr_text']}, 파일명: {info['scene_id']}\n"

    prompt = f"""
    너는 웹툰 검색 도우미야. 아래의 [검색된 장면들]을 바탕으로 사용자 질문에 답변해줘.
    어떤 장면인지 설명하고, 해당 장면의 파일명도 언급해줘.
    
    질문: {user_query}
    
    [검색된 장면들]
    {context_str}
    
    답변:"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content