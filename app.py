# app.py (메인 애플리케이션 파일)
import os
import logging
import traceback
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from getpass import getpass
from datetime import datetime

# --- 모듈 임포트 ---
# 온톨로지
try:
    from ontologybyRAG.rdf_retriever import SimpleRdfInfoRetriever
    from ontologybyRAG.llm_handler import OpenAiHandler # 수정된 핸들러
    ONTOLOGY_AVAILABLE = True
except ImportError:
    logging.warning("온톨로지 관련 모듈(ontologybyRAG) 로드 실패. 온톨로지 검색 비활성화.")
    ONTOLOGY_AVAILABLE = False
    SimpleRdfInfoRetriever = None # 클래스 정의 없음
    OpenAiHandler = None # 아래에서 정의하므로 괜찮음

# 벡터 (Langchain 필요 여부 확인)
try:
    from vectorbyRAG.config import ROOT_FOLDER_PATH, SEARCH_K # 설정값 임포트
    from vectorbyRAG.document_processor import DocumentProcessor
    from vectorbyRAG.vector_store_manager import VectorStoreManager
    VECTOR_AVAILABLE = True
except ImportError:
    logging.warning("벡터 관련 모듈(vectorbyRAG) 또는 Langchain 로드 실패. 벡터 검색 비활성화.")
    VECTOR_AVAILABLE = False
    DocumentProcessor = None
    VectorStoreManager = None
    ROOT_FOLDER_PATH = None # 경로 설정도 의미 없음
    SEARCH_K = 5 # 기본값

# --- OpenAiHandler 클래스 정의 (import 실패 시 대체용 또는 llm_handler.py가 없을 경우) ---
if OpenAiHandler is None:
    from openai import OpenAI
    logger_llm = logging.getLogger(__name__)
    class OpenAiHandler: # 위 llm_handler.py 내용과 동일하게 정의
        def __init__(self, api_key: str):
            try: self.client = OpenAI(api_key=api_key); logger_llm.info("OpenAI 클라이언트 초기화 성공 (대체).")
            except Exception as e: logger_llm.error(f"OpenAI 클라이언트 초기화 실패 (대체): {e}"); self.client = None; raise
        def generate_answer_with_history(self, system_prompt: str, history: list[dict], current_context: str, current_user_message: str, model: str):
            if not self.client: logger_llm.error("OpenAI 클라이언트 미초기화 (대체)."); return None
            messages_for_api = [{"role": "system", "content": system_prompt}]
            messages_for_api.extend(history)
            final_user_content = f"...\n\n### 현재 검색된 정보 (컨텍스트):\n{current_context}\n-----\n### 현재 질문:\n{current_user_message}" # 포맷팅 주의
            messages_for_api.append({"role": "user", "content": final_user_content})
            try:
                completion = self.client.chat.completions.create(model=model, messages=messages_for_api, temperature=0.3)
                return completion.choices[0].message.content.strip()
            except Exception as e: logger_llm.error(f"OpenAI API 호출 오류 (대체): {e}"); return None


# --- 로깅 및 설정 (이전 코드와 유사) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
logger = logging.getLogger(__name__) # 메인 앱 로거

RDF_FILE = "./ontologybyRAG/05-02.rdf" # 온톨로지 파일 경로
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_GENERATION_MODEL = "gpt-4.1-mini-2025-04-14"
MAX_HISTORY_LENGTH = 10

# --- 전역 변수 ---
rdf_retriever : SimpleRdfInfoRetriever | None = None
doc_processor : DocumentProcessor | None = None
vector_manager: VectorStoreManager | None = None
llm_handler : OpenAiHandler | None = None
initialization_error : str | None = None
conversation_history : list[dict] = []


# --- 온톨로지 정보 포매팅 함수 (기존 코드 유지) ---
def format_rdf_info_for_llm(person_name, info_dict):
    if not info_dict or (not info_dict.get("direct") and not info_dict.get("inverse")):
        return f"{person_name}에 대한 정보가 온톨로지 그래프에 없습니다."
    # ... (이전 코드와 동일한 포매팅 로직) ...
    lines = [f"--- {person_name} 관련 온톨로지 정보 ---"]
    # (직접 정보 포매팅)
    if info_dict.get("direct"): lines.append("\n[직접 속성]"); lines.extend([f"- {item['predicate']}: {item['object']}" for item in sorted(info_dict["direct"], key=lambda x: x['predicate'])])
    # (연관 항목 포매팅)
    if info_dict.get("inverse"):
        lines.append("\n[연관 항목]")
        for item in sorted(info_dict["inverse"], key=lambda x: x['label']):
            details_text = "\n  - " + "\n  - ".join(f"{prop}: {val}" for prop, val in sorted(item['details'])) if item['details'] else ""
            lines.append(f"\n* {item['label']} ({item['relation_to_person']}){details_text}")
    return "\n".join(lines)

# --- 통합 시스템 초기화 함수 (모듈화된 버전) ---
def initialize_hybrid_system():
    global rdf_retriever, doc_processor, vector_manager, llm_handler, initialization_error
    try:
        logger.info("하이브리드 RAG 시스템 초기화 시작...")
        # 0. API 키 확인 및 LLM 핸들러 초기화
        api_key = OPENAI_API_KEY
        if not api_key:
            logger.warning("OPENAI_API_KEY 환경 변수 없음. API 키를 입력하세요.")
            api_key = getpass("OpenAI API 키: ")
            if not api_key: raise ValueError("OpenAI API 키 필요.")
            # os.environ["OPENAI_API_KEY"] = api_key # 임시 설정 (권장 X)

        llm_handler = OpenAiHandler(api_key) # llm_handler.py에서 로드된 클래스 사용

        # 1. 온톨로지 리트리버 초기화
        if ONTOLOGY_AVAILABLE and SimpleRdfInfoRetriever:
            abs_rdf_path = os.path.abspath(RDF_FILE)
            if os.path.exists(abs_rdf_path):
                rdf_retriever = SimpleRdfInfoRetriever(abs_rdf_path)
                logger.info("RDF 리트리버 초기화 완료.")
            else:
                logger.warning(f"RDF 파일({abs_rdf_path}) 없음. 온톨로지 검색 비활성화.")
        else:
             logger.warning("온톨로지 모듈 사용 불가.")


        # 2. 벡터 저장소 초기화
        if VECTOR_AVAILABLE and DocumentProcessor and VectorStoreManager:
            doc_processor = DocumentProcessor()
            vector_manager = VectorStoreManager(api_key) # VectorStoreManager에 api_key 전달

            # 파일 로드 및 청킹
            processed_docs = doc_processor.load_and_process_pdfs(ROOT_FOLDER_PATH)
            chunks = doc_processor.split_documents(processed_docs)

            # 벡터 저장소 생성/로드
            if not vector_manager.create_or_load_store(chunks):
                logger.error("벡터 저장소 생성/로드 실패. 벡터 검색 비활성화.")
                vector_manager = None # 실패 시 비활성화
            else:
                 logger.info("벡터 저장소 초기화 완료.")
        else:
            logger.warning("벡터 모듈 또는 Langchain 사용 불가.")


        logger.info("하이브리드 RAG 시스템 초기화 완료.")

    except Exception as e:
        logger.exception("시스템 초기화 중 치명적 오류 발생")
        initialization_error = f"초기화 실패: {e}"
        
# --- Flask 앱 설정 ---
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# 앱 실행 전에 시스템 초기화
initialize_hybrid_system()

@app.route("/")
def index():
    global conversation_history # 페이지 로드 시 대화 기록 초기화
    conversation_history = []
    logging.info("새로운 세션 시작: 대화 기록 초기화됨.")

    if initialization_error:
        return render_template('error.html', error_message=f"시스템 초기화 실패: {initialization_error}")
    suggested_questions = [
        "정진원이 쓴 논문과 관련 파일 내용은?",
        "SAR 선박 탐지 관련 논문의 요약과 파일 위치 알려줘.",
        "정진원의 특허 정보와 관련 출원 서류 파일 찾아줘.",
        "Extended U-Net 논문의 저자는 누구야?",
        "정진원의 이메일 주소는 뭐야?"
    ]
    return render_template('index.html', suggestions=suggested_questions)


# --- 채팅 처리를 위한 API 엔드포인트 (모듈화된 버전) ---
@app.route("/chat", methods=['POST'])
def chat():
    global conversation_history

    if initialization_error: return jsonify({'error': f"시스템 오류: {initialization_error}"}), 500
    if not llm_handler: return jsonify({'error': "LLM 핸들러 미초기화."}), 500

    try: # 요청 처리
        data = request.get_json(); user_message = data['message'].strip()
        if not data or 'message' not in data or not user_message: return jsonify({'error': "'message' 필드 필요/내용 없음."}), 400
        logger.info(f"API 수신 메시지: {user_message}")
    except Exception as e: return jsonify({'error': f"잘못된 요청 형식: {e}"}), 400

    response_message = ""
    status_code = 200

    try:
        # --- 하이브리드 RAG 로직 ---
        # 1. 온톨로지 검색
        ontology_context_str = "온톨로지 정보 없음."
        if rdf_retriever:
            try:
                person_uri, person_name = rdf_retriever.find_person_uri(user_message)
                if person_uri and person_name:
                    all_info = rdf_retriever.get_all_related_info(person_uri)
                    ontology_context_str = format_rdf_info_for_llm(person_name, all_info)
                    logger.info(f"온톨로지 검색 완료 (대상: {person_name})")
                else: ontology_context_str = "질문 내 특정 인물 미발견 (온톨로지)."
            except Exception as e: logger.error(f"온톨로지 검색 오류: {e}")
        else: logger.warning("온톨로지 리트리버 비활성.")

        # 2. 벡터 검색
        vector_context_str = "벡터 정보 없음."
        if vector_manager: # VectorStoreManager 인스턴스 사용
            try:
                relevant_docs = vector_manager.search_similar_documents(user_message) # k는 config에서 설정됨
                vector_context_str = vector_manager.format_retrieved_docs(relevant_docs) # 포매팅 함수 호출
                logger.info(f"벡터 검색 완료 ({len(relevant_docs)}개 문서)")
            except Exception as e: logger.error(f"벡터 검색 오류: {e}")
        else: logger.warning("벡터 저장소 비활성.")

        # 3. 현재 컨텍스트 융합
        current_combined_context = f"""\
### 정보 소스 1: 온톨로지 그래프
{ontology_context_str}

### 정보 소스 2: 관련 파일 내용
{vector_context_str}
"""

        # 4. LLM 답변 생성 (대화 기록 포함)
        system_prompt = """
당신은 AI 어시스턴트입니다. 온톨로지 정보, 파일 내용, 그리고 이전 대화 내용을 종합하여 답변합니다.
- 파일 위치나 내용을 물으면 '정보 소스 2'의 출처 정보를 활용하세요.
- 정보가 부족하면 솔직하게 말하세요.
- "hasFavoriteFoodName", "hasFavoritePersonName" 정보는 직접 언급되지 않는 이상 답변에 사용하지 마세요.
"""
        # 대화 기록 길이 제한
        if len(conversation_history) >= MAX_HISTORY_LENGTH * 2:
            conversation_history = conversation_history[-(MAX_HISTORY_LENGTH * 2 - 2):]

        llm_answer = llm_handler.generate_answer_with_history(
            system_prompt=system_prompt,
            history=conversation_history,
            current_context=current_combined_context,
            current_user_message=user_message,
            model=LLM_GENERATION_MODEL
        )

        if llm_answer: # 답변 성공 시
            response_message = llm_answer
            conversation_history.append({"role": "user", "content": user_message})
            conversation_history.append({"role": "assistant", "content": llm_answer})
            logger.info(f"대화 기록 업데이트됨 (현재 턴 수: {len(conversation_history)//2})")
        else: # 답변 생성 실패 시
            response_message = "답변 생성 중 오류 발생."
            status_code = 500

    except Exception as e: # 전체 RAG 로직 오류 처리
        logger.exception("메시지 처리 중 예상치 못한 오류 발생")
        response_message = f"죄송합니다. 서버 내부 오류: {e}"
        status_code = 500

    # 최종 응답 반환
    if status_code == 200: return jsonify({'response': response_message}), status_code
    else: return jsonify({'error': response_message}), status_code


# --- 메인 실행 ---
if __name__ == "__main__":
    logger.info("Flask 애플리케이션 시작 준비...")
    # API 키 관련 로직은 초기화 함수 내부에 있음
    print("\n하이브리드 RAG Flask 웹 서버 (모듈화 + 대화 기록) 시작합니다 (http://localhost:5000)")
    print("종료하려면 Ctrl+C를 누르세요.\n")
    app.run(debug=True, host='0.0.0.0', port=5000)