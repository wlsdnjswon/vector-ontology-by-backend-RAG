# vectorbyRAG/document_processor.py
import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List # Import List for type hinting

# 설정값 가져오기
from .config import CHUNK_SIZE, CHUNK_OVERLAP

# 로깅 설정
logger = logging.getLogger(__name__) # Use the same logger configured in app.py

class DocumentProcessor:
    """
    파일 시스템에서 문서를 로드하고, 처리하며, 각 청크에 고유 ID를 부여하는 클래스.
    VectorStoreManager에서 업데이트 로직을 사용하기 위해 'doc_id'를 생성합니다.
    """

    def __init__(self):
        """초기화합니다. 텍스트 스플리터를 인스턴스 변수로 생성합니다."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""], # 구분자 유지
            add_start_index=False, # 시작 인덱스는 ID 생성에 사용하지 않음
        )
        logger.info(f"DocumentProcessor 초기화됨 (Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})")

    def load_and_process_pdfs(self, root_directory: str) -> List[Document]:
        """
        지정된 디렉토리 및 하위 디렉토리에서 PDF 파일을 로드하고,
        페이지별로 Document 객체를 생성하며 기본적인 메타데이터를 추가합니다.

        Args:
            root_directory (str): 검색을 시작할 루트 디렉토리 경로.

        Returns:
            List[Document]: 로드된 각 페이지에 대한 Document 객체 리스트.
                            각 Document는 'filename', 'full_path', 'category',
                            'author', 'page' 메타데이터를 포함합니다.
        """
        logger.info(f"'{root_directory}'에서 PDF 로딩 및 기본 메타데이터 처리 시작...")
        all_page_docs: List[Document] = []
        abs_root_directory = os.path.abspath(root_directory) # 절대 경로 사용

        if not os.path.exists(abs_root_directory):
            logger.error(f"지정된 루트 디렉토리를 찾을 수 없습니다: '{abs_root_directory}'")
            return all_page_docs # 빈 리스트 반환

        for dirpath, _, filenames in os.walk(abs_root_directory):
            pdf_files = [f for f in filenames if f.lower().endswith(".pdf")]
            if not pdf_files:
                continue # PDF 파일 없으면 다음 디렉토리로

            logger.debug(f"'{dirpath}' 디렉토리에서 PDF {len(pdf_files)}개 발견.")
            for filename in pdf_files:
                file_path = os.path.join(dirpath, filename)
                try:
                    loader = PyPDFLoader(file_path)
                    # loader.load()는 페이지별 Document 리스트를 반환하며,
                    # 각 Document는 이미 'source' (file_path)와 'page' 메타데이터를 가짐
                    pages: List[Document] = loader.load()

                    if not pages:
                         logger.warning(f"PDF 파일 로드 결과 페이지 없음: {file_path}")
                         continue

                    # 추가 메타데이터 부여 (기존 메타데이터에 업데이트)
                    relative_path = os.path.relpath(dirpath, abs_root_directory)
                    path_parts = relative_path.split(os.sep)
                    # 카테고리: 루트 바로 아래 폴더명 (없으면 'Root')
                    category = path_parts[0] if path_parts and path_parts[0] != '.' else "Root"
                    # 저자: 루트 폴더명 (예시) - 필요시 다른 로직 적용 가능
                    author = os.path.basename(abs_root_directory)

                    for page_doc in pages:
                        # PyPDFLoader가 추가한 기본 메타데이터 ('source', 'page') 확인 및 활용
                        if 'source' not in page_doc.metadata:
                             page_doc.metadata['source'] = file_path # 만약 없다면 추가
                        if 'page' not in page_doc.metadata:
                             logger.warning(f"'page' 메타데이터 누락: {file_path} - 페이지 번호 불명확")
                             # 페이지 번호가 없으면 ID 생성에 문제 생길 수 있음

                        # 커스텀 메타데이터 업데이트
                        page_doc.metadata.update({
                            'filename': filename,
                            'full_path': file_path, # source와 중복될 수 있으나 명시적 추가
                            'category': category,
                            'author': author
                        })
                    all_page_docs.extend(pages)
                    logger.debug(f"  - 로드 및 메타데이터 처리 완료: {file_path} ({len(pages)} 페이지)")

                except FileNotFoundError:
                     logger.error(f"PDF 파일 접근 불가 (삭제되었거나 권한 문제?): {file_path}")
                except Exception as e:
                    # 구체적인 오류 로깅 (예: 암호화된 PDF 등)
                    logger.warning(f"PDF 로딩 중 예외 발생 (파일 건너뜀): {file_path} - {type(e).__name__}: {e}", exc_info=False) # 스택 트레이스 제외

        logger.info(f"총 {len(all_page_docs)} 페이지(Document) 로드 및 기본 메타데이터 처리 완료.")
        return all_page_docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        주어진 Document 리스트(페이지 단위)를 설정된 크기의 청크로 분할하고,
        각 청크에 고유하고 안정적인 'doc_id' 메타데이터를 추가합니다.

        'doc_id' 형식: "<full_path>::page_<page_number>::chunk_<chunk_index_within_page>"

        Args:
            documents (List[Document]): `load_and_process_pdfs`에서 반환된
                                        페이지 단위 Document 리스트.

        Returns:
            List[Document]: 청크로 분할되고 각 청크에 'doc_id'가 추가된
                            Document 리스트.
        """
        if not documents:
            logger.warning("분할할 문서(페이지)가 없습니다.")
            return []

        logger.info(f"텍스트 청크 분할 및 'doc_id' 생성 시작 (총 {len(documents)} 페이지 대상)...")
        all_chunks: List[Document] = []
        missing_metadata_count = 0

        for doc_index, page_doc in enumerate(documents):
            # ID 생성에 필요한 메타데이터 확인
            full_path = page_doc.metadata.get('full_path')
            page_number = page_doc.metadata.get('page') # PyPDFLoader가 0부터 시작하는 정수 추가

            if full_path is None or page_number is None:
                logger.warning(f"{doc_index}번째 문서(페이지)에 'full_path' 또는 'page' 메타데이터 누락. "
                               f"해당 페이지에서 생성되는 청크에는 'doc_id'가 부여되지 않을 수 있습니다. "
                               f"메타데이터: {page_doc.metadata}")
                missing_metadata_count += 1
                # 이 페이지는 건너뛸 수도 있지만, 일단 청킹은 시도하고 ID만 누락되도록 함
                # 또는 여기서 continue 하여 아예 청킹 대상에서 제외할 수도 있음.

            # 개별 페이지 Document를 청크로 분할
            # create_documents는 내용 리스트와 메타데이터 리스트를 받음
            # 메타데이터가 각 청크에 복사/상속됨
            try:
                page_content = page_doc.page_content
                if not page_content.strip():
                    # logger.debug(f"페이지 내용이 비어있어 청킹 건너<0xEB><0x9B><0x84> (Path: {full_path}, Page: {page_number})")
                    continue # 내용 없으면 청크 생성 불가

                chunks_from_page: List[Document] = self.text_splitter.create_documents(
                    [page_content], # 내용을 리스트로 감싸서 전달
                    metadatas=[page_doc.metadata] # 해당 페이지의 메타데이터를 리스트로 전달
                )

                # 생성된 청크들에 대해 'doc_id' 부여
                for chunk_index, chunk in enumerate(chunks_from_page):
                    if full_path is not None and page_number is not None:
                        # 안정적인 ID 생성
                        doc_id = f"{full_path}::page_{page_number}::chunk_{chunk_index}"
                        chunk.metadata['doc_id'] = doc_id
                    else:
                        # 필수 메타데이터가 없으면 ID 생성 불가
                        chunk.metadata['doc_id'] = None # 명시적으로 None 설정 또는 키를 아예 추가 안 함
                        logger.warning(f"'doc_id' 생성 불가 (메타데이터 부족): 청크 출처 "
                                       f"({page_doc.metadata.get('filename', 'N/A')}, "
                                       f"페이지 추정: {page_number})")

                    all_chunks.append(chunk)

            except Exception as e:
                logger.error(f"페이지 청킹 중 오류 발생 (Path: {full_path}, Page: {page_number}): {e}", exc_info=True)


        final_chunk_count = len(all_chunks)
        chunks_with_id = sum(1 for chunk in all_chunks if chunk.metadata.get('doc_id') is not None)

        logger.info(f"텍스트 청크 분할 완료. 총 {final_chunk_count}개의 청크 생성됨.")
        if final_chunk_count > 0:
             logger.info(f"  - 'doc_id'가 성공적으로 부여된 청크 수: {chunks_with_id}")
             logger.debug(f"첫번째 청크 메타데이터 예시: {all_chunks[0].metadata if all_chunks else 'N/A'}")
             logger.debug(f"마지막 청크 메타데이터 예시: {all_chunks[-1].metadata if all_chunks else 'N/A'}")
        if missing_metadata_count > 0:
            logger.warning(f"{missing_metadata_count}개 페이지에서 'full_path' 또는 'page' 메타데이터가 누락되어 "
                           f"일부 청크({final_chunk_count - chunks_with_id}개 추정)에 'doc_id'가 부여되지 않았을 수 있습니다.")


        return all_chunks