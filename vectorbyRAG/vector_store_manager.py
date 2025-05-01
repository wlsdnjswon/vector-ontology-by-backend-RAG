# vectorbyRAG/vector_store_manager.py
import os
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
# 설정값 가져오기
from .config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    LLM_EMBEDDING_MODEL,
    SEARCH_K
)

# 로깅 설정
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """벡터 저장소(ChromaDB)를 관리하고 검색 기능을 제공하는 클래스"""

    def __init__(self, api_key: str):
        """임베딩 모델을 초기화합니다."""
        try:
            self.embeddings = OpenAIEmbeddings(
                model=LLM_EMBEDDING_MODEL,
                api_key=api_key
            )
            self.vector_store = None
            logger.info("OpenAI 임베딩 모델 초기화 완료.")
        except Exception as e:
            logger.error(f"OpenAI 임베딩 모델 초기화 실패: {e}")
            self.embeddings = None
            self.vector_store = None
            raise # 초기화 실패는 심각한 문제일 수 있으므로 에러 발생

    def create_or_load_store(self, chunks: list[Document]):
        """ChromaDB 벡터 저장소를 생성하거나 로드합니다."""
        if not self.embeddings:
            logger.error("임베딩 모델이 초기화되지 않아 벡터 저장소를 생성/로드할 수 없습니다.")
            return False

        try:
            if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
                logger.info(f"기존 ChromaDB 로딩 시도: {CHROMA_PERSIST_DIR}")
                self.vector_store = Chroma(
                    persist_directory=CHROMA_PERSIST_DIR,
                    embedding_function=self.embeddings,
                    collection_name=COLLECTION_NAME
                )
                logger.info("ChromaDB 로딩 완료.")
            else:
                logger.info("새로운 ChromaDB 생성 시도...")
                if not chunks:
                    logger.warning("벡터 저장소를 생성할 청크가 없습니다.")
                    # 청크가 없어도 빈 DB를 생성할 수 있게 할지, 에러를 낼지 결정 필요
                    # 여기서는 빈 DB를 생성하지 않고 로드만 시도하는 형태로 유지
                    # 또는 Chroma 인스턴스만 생성하고 나중에 add_documents 호출
                    # self.vector_store = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=self.embeddings, collection_name=COLLECTION_NAME)
                    # logger.warning("청크가 없어 비어있는 ChromaDB 인스턴스만 생성됨 (저장 X)")
                    # -> from_documents를 사용해야 저장이 되므로, 청크가 없을 때 생성 자체를 막는 것이 나을 수 있음
                    logger.error("새로운 DB를 생성할 청크가 제공되지 않았습니다.")
                    return False # 생성 실패

                self.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=CHROMA_PERSIST_DIR, # 생성 시 자동 저장
                    collection_name=COLLECTION_NAME
                )
                logger.info(f"ChromaDB 생성 및 저장 완료: {CHROMA_PERSIST_DIR}")
            return True # 성공적으로 로드 또는 생성됨
        except Exception as e:
            logger.error(f"ChromaDB 생성/로드 중 오류 발생: {e}")
            self.vector_store = None # 오류 발생 시 저장소 참조 제거
            return False # 실패

    def search_similar_documents(self, query: str, k: int = SEARCH_K) -> list[Document]:
        """주어진 쿼리와 유사한 문서를 벡터 저장소에서 검색합니다."""
        if not self.vector_store:
            logger.warning("벡터 저장소가 초기화되지 않았거나 사용할 수 없어 검색을 건너<0xEB><0x9B><0x84>니다.")
            return []
        try:
            logger.debug(f"벡터 유사도 검색 수행: query='{query}', k={k}")
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"유사도 검색 완료: {len(results)}개 문서 반환됨.")
            return results
        except Exception as e:
            logger.error(f"유사도 검색 중 오류 발생: {e}")
            return []

    def format_retrieved_docs(self, docs: list[Document]) -> str:
        """검색된 문서 리스트를 LLM 컨텍스트용 문자열로 포매팅합니다."""
        if not docs:
            return "관련 파일 내용을 찾지 못했습니다."

        formatted_lines = ["--- 관련 파일 내용 및 출처 ---"]
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            source_info = (
                f"출처 {i+1}: [파일명: {metadata.get('filename', 'N/A')}, "
                f"경로: {metadata.get('full_path', 'N/A')}, "
                f"페이지: {metadata.get('page', 'N/A')}, "
                f"카테고리: {metadata.get('category', 'N/A')}]"
                # 필요시 작성자 메타데이터도 추가
                # f", 작성자: {metadata.get('author', 'N/A')}]"
            )
            # 내용을 적절히 줄여서 포함 (너무 길면 컨텍스트 제한 초과 가능)
            content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            formatted_lines.append(f"{source_info}\n내용 요약: {content_preview}\n---")

        return "\n".join(formatted_lines)