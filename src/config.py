# src/config.py
import yaml
import os

class Config:
    def __init__(self, config_path="./config/config.yaml"):
        # 파일이 없으면 에러 발생
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
            
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    # .yaml 파일의 계층 구조를 속성(@property)으로 접근 가능하게 함
    @property
    def paths(self): 
        return self.cfg["paths"]
    
    @property
    def models(self): 
        return self.cfg["models"]
    
    @property
    def rag(self): 
        return self.cfg["rag"]

# 전역에서 사용할 싱글톤 객체 생성
settings = Config()