# src/modules/rewriter.py
from openai import OpenAI
from config import OPENAI_MODEL

client = OpenAI() # API Key는 .env에서 자동 로드

def rewrite_query(user_query: str) -> str:
    # 간단한 프롬프트 엔지니어링
    prompt = f"""
    사용자의 질문을 웹툰 장면 검색에 적합한 구체적인 문장으로 수정해줘.
    명사 위주로, 시각적 묘사를 포함해서 한 문장으로 만들어.
    질문: {user_query}
    수정된 질문:"""
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except:
        return user_query # 에러나면 원본 사용