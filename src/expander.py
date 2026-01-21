#src/expander.py
import json
import os
from src.config import settings
from src.prompts import REWRITE_SYSTEM 

class QueryExpander:
    def __init__(self, llm):
        self.llm = llm
        
        char_path = os.path.join(settings.paths["data_dir"], "characters.json")
        self.characters = [] 
        self.char_context = "" 
        self._load_characters(char_path)

    def _load_characters(self, path):
        if not os.path.exists(path):
            print(f"⚠️ 캐릭터 파일 없음: {path}")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.characters = data.get('characters', data)

        # 1. Rewriter용 매핑 정보 생성 (모든 후보 포함)
        mapping_list = []
        for char in self.characters:
            all_names = char.get('name_candidates', [])
            if not all_names: continue
            
            # 대표 이름 (첫 번째)
            primary_name = all_names[0]
            
            # 나머지 별명들 (순서 유지하며 중복 제거)
            aliases = [n for n in all_names if n != primary_name]
            alias_str = ", ".join(aliases)
            
            # 감지할 키워드들
            keywords = ", ".join(all_names)
            
            # ★ 핵심: LLM에게 "이 키워드가 보이면 -> 대표이름(별명1, 별명2...)"로 바꾸라고 지시
            line = f"- 감지 키워드: [{keywords}] -> 변환: {primary_name}({alias_str})"
            mapping_list.append(line)
        
        self.char_context = "\n".join(mapping_list)

    def expand(self, query):
        if not self.characters:
            return query

        system_prompt = REWRITE_SYSTEM.format(char_list=self.char_context)
        
        rewritten_query = self.llm.ask(system_prompt, f"User: {query}")
        
        clean_query = rewritten_query.replace("Rewritten:", "").strip().strip('"').strip("'")
        return clean_query

    def get_profile_str(self, query):
        """답변 생성용 프로필 추출 (검색된 모든 후보 이름 활용)"""
        profiles = []
        for char in self.characters:
            candidates = char.get('name_candidates', [])
            if not candidates: continue

            # 확장된 쿼리(search_query)에는 모든 이름이 다 들어있을 테니 매칭 확률 매우 높음
            if any(name in query for name in candidates):
                
                # 모든 이름 나열
                all_names_str = ", ".join(candidates)
                
                app_list = char.get('appearance_traits', [])
                app_str = ", ".join(app_list) if app_list else "정보 없음"
                
                beh_list = char.get('behavior_traits', [])
                beh_str = "\n    - ".join(beh_list) if beh_list else "정보 없음"
                
                info = (
                    f"### 인물: {all_names_str}\n" # 제목에 모든 이름 박아버림
                    f"  * [외모 특징]: {app_str}\n"
                    f"  * [성격 및 행동]:\n    - {beh_str}\n"
                )
                profiles.append(info)
                 
        if not profiles:
            return "(질문에 명시된 인물 정보가 없습니다. 문맥을 통해 파악하세요.)"
            
        return "\n".join(profiles)