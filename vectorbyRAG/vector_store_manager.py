# vectorbyRAG/vector_store_manager.py
import os
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List, Dict, Set, Optional

# 설정값 가져오기
from .config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    LLM_EMBEDDING_MODEL,
    SEARCH_K
)

# 로깅 설정
logger = logging.getLogger(__name__) # app.py와 같은 로거 사용 가능

class VectorStoreManager:
    """
    벡터 저장소(ChromaDB)를 관리하고 검색 기능을 제공하는 클래스.
    기존 저장소 로드 시 입력된 청크를 기준으로 내용을 업데이트(추가, 삭제, 수정)합니다.

    **중요:** 이 클래스는 `create_or_load_store`에 전달되는 `chunks` 리스트의
    각 `Document` 객체가 `metadata` 딕셔너리 안에 'doc_id'라는 키로
    고유하고 안정적인 ID(문자열)를 가지고 있다고 가정합니다.
    이 'doc_id'는 이 클래스 외부(예: vectorbyRAG.DocumentProcessor)에서
    문서 로딩 및 청킹 과정 중에 생성되어야 합니다.
    """

    def __init__(self, api_key: str):
        """OpenAI 임베딩 모델을 초기화합니다."""
        self.vector_store: Optional[Chroma] = None
        self.embeddings: Optional[OpenAIEmbeddings] = None
        try:
            self.embeddings = OpenAIEmbeddings(
                model=LLM_EMBEDDING_MODEL,
                api_key=api_key
            )
            logger.info(f"OpenAI 임베딩 모델 ({LLM_EMBEDDING_MODEL}) 초기화 완료.")
        except Exception as e:
            logger.error(f"OpenAI 임베딩 모델 초기화 실패: {e}", exc_info=True)
            # app.py의 initialize_hybrid_system에서 이 예외를 처리할 수 있도록 raise
            raise

    def _validate_and_extract_ids(self, chunks: List[Document]) -> Optional[Dict[str, Document]]:
        """
        입력된 청크 리스트를 검증하고, 각 청크에서 'doc_id'를 추출하여 맵을 생성합니다.
        하나라도 'doc_id'가 없거나 유효하지 않으면 None을 반환하고 에러를 로깅합니다.
        """
        if not chunks:
            logger.info("입력된 청크가 없습니다. ID 추출을 건너<0xEB><0x9B><0x84>니다.")
            return {} # 빈 딕셔너리 반환

        chunk_map: Dict[str, Document] = {}
        invalid_chunks_count = 0
        source_example = "N/A" # 오류 로깅 시 예시 경로

        for i, chunk in enumerate(chunks):
            if chunk.metadata:
                 source_example = chunk.metadata.get('full_path', chunk.metadata.get('filename', 'N/A'))

            doc_id = chunk.metadata.get('doc_id')

            if doc_id is None:
                logger.error(f"오류: {i}번째 청크 (소스 추정: {source_example})에 'doc_id' 메타데이터 키가 없습니다! "
                             f"DocumentProcessor에서 'doc_id'를 생성해야 합니다.")
                invalid_chunks_count += 1
            elif not isinstance(doc_id, str) or not doc_id.strip():
                 logger.error(f"오류: {i}번째 청크 (소스 추정: {source_example})의 'doc_id'가 유효한 문자열이 아닙니다: '{doc_id}'.")
                 invalid_chunks_count += 1
            elif doc_id in chunk_map:
                logger.warning(f"경고: 중복된 'doc_id' ('{doc_id}') 발견됨. "
                               f"이전 청크 소스: {chunk_map[doc_id].metadata.get('full_path', 'N/A')}, "
                               f"현재 청크 소스: {source_example}. "
                               f"현재 청크로 덮어쓰기됩니다.")
                # 중복 ID 처리: 여기서는 경고 후 덮어쓰기
                chunk_map[doc_id] = chunk
            else:
                 chunk_map[doc_id] = chunk

        if invalid_chunks_count > 0:
            logger.error(f"총 {invalid_chunks_count}개의 청크에서 'doc_id'가 없거나 유효하지 않습니다. "
                         "이 문제는 vectorbyRAG.DocumentProcessor 또는 해당 기능을 수행하는 곳에서 "
                         "각 Document의 metadata에 고유하고 안정적인 'doc_id'를 추가하여 해결해야 합니다. "
                         "벡터 저장소 업데이트를 진행할 수 없습니다.")
            return None # 검증 실패

        logger.info(f"입력된 {len(chunks)}개 청크에서 고유 ID {len(chunk_map)}개를 성공적으로 추출했습니다.")
        return chunk_map


    def create_or_load_store(self, chunks: List[Document]) -> bool:
        """
        ChromaDB 벡터 저장소를 생성하거나 로드합니다. 로드 시, 입력된 `chunks`와
        기존 저장소의 내용을 비교하여 변경 사항(추가, 삭제, 업데이트)을 적용합니다.

        Args:
            chunks (List[Document]): `DocumentProcessor` 등에서 생성된 현재 문서 청크 리스트.
                                      **각 Document는 metadata에 고유하고 안정적인 'doc_id'(문자열)를
                                      반드시 포함해야 합니다.**

        Returns:
            bool: 성공 여부 (True: 성공, False: 실패). app.py에서 이 값을 확인합니다.
        """
        if not self.embeddings:
            logger.error("임베딩 모델이 초기화되지 않아 벡터 저장소를 생성/로드할 수 없습니다.")
            return False

        # 1. 입력 청크 검증 및 ID 추출
        current_chunk_map = self._validate_and_extract_ids(chunks)
        if current_chunk_map is None: # 검증 실패 시 (오류는 _validate_and_extract_ids 내부에서 로깅됨)
            return False
        current_ids_set: Set[str] = set(current_chunk_map.keys())

        try:
            db_path = os.path.abspath(CHROMA_PERSIST_DIR) # 절대 경로 사용
            db_exists = os.path.exists(db_path) and os.listdir(db_path)

            if db_exists:
                # --- 2a. 기존 DB 로드 ---
                logger.info(f"기존 ChromaDB 로딩 시도: {db_path}")
                self.vector_store = Chroma(
                    persist_directory=db_path,
                    embedding_function=self.embeddings,
                    collection_name=COLLECTION_NAME
                )
                logger.info(f"컬렉션 '{COLLECTION_NAME}' 로딩 완료.")

                # --- 3. 변경 사항 감지 및 적용 ---
                logger.info("기존 DB와 현재 청크 비교 및 동기화 시작...")
                try:
                    existing_docs_data = self.vector_store.get(include=["metadatas"]) # IDs와 Metadatas 가져오기
                    existing_ids_set: Set[str] = set(existing_docs_data.get('ids', []))
                    logger.info(f"기존 DB에 문서 ID {len(existing_ids_set)}개 확인됨.")
                except Exception as e:
                    logger.error(f"기존 DB에서 ID 목록을 가져오는 중 오류 발생: {e}", exc_info=True)
                    self.vector_store = None # 로드 실패 간주
                    return False # ID 비교 불가 -> 동기화 불가

                # --- 삭제 처리 ---
                ids_to_delete: List[str] = list(existing_ids_set - current_ids_set)
                if ids_to_delete:
                    logger.info(f"삭제할 문서 ID {len(ids_to_delete)}개 식별됨.")
                    try:
                        self.vector_store.delete(ids=ids_to_delete)
                        logger.info(f"문서 {len(ids_to_delete)}개 삭제 완료.")
                    except Exception as e:
                        logger.error(f"DB에서 문서 삭제 중 오류 발생: {e}", exc_info=True)
                        # 삭제 실패 시 일단 계속 진행 (다음 단계 시도)
                else:
                    logger.info("삭제할 문서 없음 (기존 ID가 모두 현재 ID에 포함됨).")

                # --- 추가 처리 ---
                ids_to_add: List[str] = list(current_ids_set - existing_ids_set)
                if ids_to_add:
                    logger.info(f"추가할 문서 ID {len(ids_to_add)}개 식별됨.")
                    chunks_to_add: List[Document] = [current_chunk_map[add_id] for add_id in ids_to_add]
                    try:
                        self.vector_store.add_documents(documents=chunks_to_add, ids=ids_to_add)
                        logger.info(f"신규 문서 {len(chunks_to_add)}개 추가 완료.")
                    except Exception as e:
                         logger.error(f"DB에 신규 문서 추가 중 오류 발생: {e}", exc_info=True)
                         # 추가 실패 시 일단 계속 진행
                else:
                    logger.info("새로 추가할 문서 없음 (현재 ID가 모두 기존 ID에 포함됨).")

                # --- 업데이트(Upsert) 처리 ---
                ids_to_potentially_update: List[str] = list(current_ids_set.intersection(existing_ids_set))
                if ids_to_potentially_update:
                    logger.info(f"업데이트(Upsert) 대상 문서 ID {len(ids_to_potentially_update)}개 식별됨.")
                    chunks_to_update: List[Document] = [current_chunk_map[upd_id] for upd_id in ids_to_potentially_update]
                    try:
                        # 최신 Chroma는 add_documents가 upsert (update or insert) 역할을 함
                        # ID가 존재하면 내용을 업데이트하고, 없으면 새로 추가함.
                        # 따라서 이전에 추가되지 않은 ID가 여기 포함되어도 문제 없음.
                        self.vector_store.add_documents(documents=chunks_to_update, ids=ids_to_potentially_update)
                        logger.info(f"기존 문서 {len(chunks_to_update)}개에 대한 업데이트/Upsert 시도 완료.")
                    except Exception as e:
                        logger.error(f"DB 문서 업데이트(Upsert) 중 오류 발생: {e}", exc_info=True)
                        # 업데이트 실패 시 일단 계속 진행
                else:
                    logger.info("업데이트(Upsert) 대상 문서 없음 (기존과 현재 ID의 교집합 없음).")

                # 참고: ChromaDB의 persist()는 명시적으로 호출하지 않아도 될 수 있음 (버전 확인)
                # self.vector_store.persist()
                # logger.info("DB 변경사항 영구 저장 시도 완료.")

            else:
                # --- 2b. 새로운 DB 생성 ---
                logger.info(f"기존 ChromaDB 없음. '{db_path}' 경로에 새로운 DB 생성 시도...")
                if not current_chunk_map: # ID 추출 후 청크가 없는 경우
                    logger.warning("새로운 DB를 생성해야 하지만, 유효한 청크('doc_id' 포함)가 없습니다. "
                                   "빈 컬렉션을 생성하거나 아무 작업도 하지 않을 수 있습니다. "
                                   "여기서는 작업을 중단하지 않고 진행합니다 (검색 시 결과 없음).")
                    # 빈 DB를 명시적으로 생성해야 할 수도 있음, Chroma 동작 방식 확인 필요.
                    # 일단 빈 DB를 가정하고 성공으로 간주.
                    # self.vector_store = Chroma(...) # 빈 컬렉션 생성 코드?
                else:
                    logger.info(f"{len(current_chunk_map)}개의 고유 ID를 가진 청크로 새 DB 생성 시작...")
                    self.vector_store = Chroma.from_documents(
                        documents=list(current_chunk_map.values()),
                        embedding=self.embeddings,
                        ids=list(current_chunk_map.keys()), # 생성 시 ID 필수 제공
                        persist_directory=db_path,
                        collection_name=COLLECTION_NAME
                    )
                    logger.info(f"ChromaDB 생성 및 저장 완료: {db_path}, 컬렉션: '{COLLECTION_NAME}'")

            # 모든 과정이 예외 없이 완료되면 성공
            return True

        except Exception as e:
            logger.error(f"ChromaDB 생성/로드/업데이트 중 예기치 않은 최상위 오류 발생: {e}", exc_info=True)
            self.vector_store = None # 오류 발생 시 저장소 참조 확실히 제거
            return False # 최종 실패 반환

    # --- search_similar_documents 와 format_retrieved_docs 는 이전과 동일 ---

    def search_similar_documents(self, query: str, k: int = SEARCH_K) -> List[Document]:
        """주어진 쿼리와 유사한 문서를 벡터 저장소에서 검색합니다."""
        if not self.vector_store:
            logger.warning("벡터 저장소가 사용 불가능하여 검색을 건너<0xEB><0x9B><0x84>니다.")
            return []
        if not self.embeddings:
             logger.warning("임베딩 모델이 초기화되지 않아 검색을 수행할 수 없습니다.")
             return []

        try:
            logger.debug(f"벡터 유사도 검색 수행: query='{query[:50]}...', k={k}")
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"유사도 검색 완료: {len(results)}개 문서 반환됨.")
            return results
        except Exception as e:
            logger.error(f"유사도 검색 중 오류 발생: {e}", exc_info=True)
            return []

    def format_retrieved_docs(self, docs: List[Document]) -> str:
        """검색된 문서 리스트를 LLM 컨텍스트용 문자열로 포매팅합니다."""
        if not docs:
            return "관련 파일 내용을 찾지 못했습니다."

        formatted_lines = ["--- 관련 파일 내용 및 출처 ---"]
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            doc_id_info = f", ID: {metadata.get('doc_id', 'N/A')}" if 'doc_id' in metadata else ""
            page_info = f"페이지: {metadata['page']}, " if metadata.get('page') is not None else ""
            source_info = (
                f"출처 {i+1}: [파일명: {metadata.get('filename', 'N/A')}, "
                f"경로: {metadata.get('full_path', 'N/A')}, "
                f"{page_info}"
                f"카테고리: {metadata.get('category', 'N/A')}"
                f"{doc_id_info}]"
            )
            content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            formatted_lines.append(f"{source_info}\n내용 요약: {content_preview}\n---")

        return "\n".join(formatted_lines)