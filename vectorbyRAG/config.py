# vectorbyRAG/config.py
import os

# --- 파일 시스템 경로 ---
# ROOT_FOLDER_PATH: PDF 파일이 있는 루트 폴더 경로 (app.py 기준 상대 경로)
ROOT_FOLDER_PATH = './정진원'

# --- ChromaDB 설정 ---
# CHROMA_PERSIST_DIR: ChromaDB 데이터를 저장할 디렉토리 경로
CHROMA_PERSIST_DIR = "./chroma_db_hybrid_hist"
# COLLECTION_NAME: ChromaDB 내에서 사용할 컬렉션 이름
COLLECTION_NAME = "hybrid_rag_hist_collection"

# --- 임베딩 및 청킹 설정 ---
# LLM_EMBEDDING_MODEL: 사용할 임베딩 모델 이름
LLM_EMBEDDING_MODEL = "text-embedding-3-small"
# CHUNK_SIZE: 텍스트를 나눌 청크 크기 (문자 수)
CHUNK_SIZE = 700
# CHUNK_OVERLAP: 청크 간 겹치는 문자 수
CHUNK_OVERLAP = 70

# --- 검색 설정 ---
# SEARCH_K: 유사도 검색 시 반환할 문서(청크) 개수
SEARCH_K = 3
