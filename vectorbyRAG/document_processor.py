# vectorbyRAG/document_processor.py
import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# 설정값 가져오기
from .config import CHUNK_SIZE, CHUNK_OVERLAP

# 로깅 설정
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """파일 시스템에서 문서를 로드하고 처리하는 클래스"""

    def __init__(self):
        # 필요시 초기 설정 추가 가능
        pass

    def load_and_process_pdfs(self, root_directory: str) -> list[Document]:
        """지정된 디렉토리 및 하위에서 PDF를 로드하고 메타데이터를 추가합니다."""
        logger.info(f"'{root_directory}'에서 PDF 로딩 및 처리 시작...")
        all_docs = []
        if not os.path.exists(root_directory):
            logger.warning(f"루트 디렉토리 '{root_directory}'를 찾을 수 없습니다.")
            return all_docs

        for dirpath, _, filenames in os.walk(root_directory):
            pdf_files = [f for f in filenames if f.lower().endswith(".pdf")]
            for filename in pdf_files:
                file_path = os.path.join(dirpath, filename)
                try:
                    loader = PyPDFLoader(file_path)
                    pages = loader.load() # 페이지별 Document 리스트 반환

                    # 메타데이터 추가
                    for page_doc in pages:
                        relative_path = os.path.relpath(dirpath, root_directory)
                        path_parts = relative_path.split(os.sep)
                        category = path_parts[0] if path_parts and path_parts[0]!='.' else "Root"
                        author = os.path.basename(root_directory) # 루트 폴더명을 저자로 간주
                        # load()가 반환하는 Document의 metadata에 직접 추가/업데이트
                        page_doc.metadata.update({
                            'filename': filename,
                            'full_path': file_path,
                            'category': category,
                            'author': author
                            # 'page' 메타데이터는 PyPDFLoader가 자동으로 추가
                        })
                    all_docs.extend(pages)
                    # logger.debug(f"  - 로드 완료: {file_path} ({len(pages)} 페이지)")
                except Exception as e:
                    logger.warning(f"PDF 로딩 오류 (파일 건너뜀): {file_path} - {e}")

        logger.info(f"총 {len(all_docs)} 페이지(Document) 로드 및 처리 완료.")
        return all_docs

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """주어진 Document 리스트를 청크로 분할합니다."""
        if not documents:
            logger.warning("분할할 문서가 없습니다.")
            return []

        logger.info("텍스트 청크 분할 시작...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""], # 구분자
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"총 {len(chunks)}개의 청크 생성 완료.")
        # logger.debug(f"첫번째 청크 메타데이터 예시: {chunks[0].metadata if chunks else 'N/A'}")
        return chunks