# src/router.py
import json
import re
from src.prompts import ROUTER_SYSTEM

class LLMRouter:
    def __init__(self, llm):
        self.llm = llm # pipeline에서 생성된 llm 인스턴스를 받음

    def route(self, user_query):
        """
        질문 -> {"intent": "...", "chapter_id": ...} 파싱
        """
        # LLM에게 물어보기
        response = self.llm.ask(ROUTER_SYSTEM, f"질문: {user_query}")
        
        # JSON 파싱 (가끔 LLM이 마크다운 ```json ... ```을 붙일 때가 있어서 제거)
        clean_json = re.sub(r"```json|```", "", response).strip()
        
        try:
            result = json.loads(clean_json)
            return result.get("intent"), result.get("chapter_id")
        except json.JSONDecodeError:
            print(f"⚠️ 라우팅 파싱 실패 (내용: {response}) -> 검색 모드로 전환")
            # 파싱 에러나면 안전하게 기본 검색으로 처리
            return "search", None