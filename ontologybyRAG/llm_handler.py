# ontologybyRAG/llm_handler.py
import os
from openai import OpenAI # OpenAI 임포트
import logging
import traceback # 상세 오류 로깅용

# 로깅 설정
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# app.py에서 이미 기본 로깅 설정을 하므로 여기서는 로거만 가져옵니다.
logger = logging.getLogger(__name__)

class OpenAiHandler:
    """OpenAI API를 사용하여 질문에 답변하고 대화 기록을 관리하는 클래스."""

    def __init__(self, api_key: str):
        """OpenAI 클라이언트를 초기화합니다."""
        if not api_key:
            logger.error("OpenAI API 키가 제공되지 않았습니다.")
            raise ValueError("OpenAI API 키가 필요합니다.")
        try:
            self.client = OpenAI(api_key=api_key)
            logger.info("OpenAI 클라이언트 초기화 성공 (llm_handler.py).")
        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 실패 (llm_handler.py): {e}")
            self.client = None
            raise

    def generate_answer_with_history(self, system_prompt: str, history: list[dict], current_context: str, current_user_message: str, model: str):
        """
        대화 기록과 현재 컨텍스트를 사용하여 답변을 생성합니다.
        이 메소드가 하이브리드 RAG 시스템의 메인 답변 생성 로직입니다.

        Args:
            system_prompt (str): LLM의 역할과 지침을 정의하는 시스템 프롬프트.
            history (list[dict]): 이전 대화 기록 ( [{'role': 'user', ...}, {'role': 'assistant', ...}] 형식).
            current_context (str): 이번 턴에 온톨로지/벡터 검색으로 얻은 컨텍스트 정보.
            current_user_message (str): 사용자의 현재 질문.
            model (str): 사용할 OpenAI 모델 이름.

        Returns:
            str: LLM이 생성한 답변 문자열, 또는 오류 발생 시 None.
        """
        if not self.client:
            logger.error("OpenAI 클라이언트가 초기화되지 않아 답변 생성을 건너<0xEB><0x9B><0x84>니다.")
            return None

        # 최종 API 호출을 위한 messages 리스트 구성
        messages_for_api = [{"role": "system", "content": system_prompt}]

        # 이전 대화 기록 추가
        messages_for_api.extend(history)

        # 현재 사용자 질문과 검색된 컨텍스트 결합 (마지막 user 메시지로 추가)
        final_user_content = f"""\
다음은 이전 대화와 현재 검색된 정보입니다. 이 정보를 바탕으로 아래 질문에 답해주세요.

### 현재 검색된 정보 (컨텍스트):
{current_context}
-----
### 현재 질문:
{current_user_message}

[답변 생성 지침]
모든 제공된 정보(이전 대화, 현재 컨텍스트)를 종합하여 현재 질문에 대한 답변을 생성해주세요.
만약 논문 관련 질문이라면 답변 끝에 '더 자세한 논문 정보는 Google Scholar 등 검색 엔진을 활용해 보세요.' 라고 덧붙여 주세요.
"""
        messages_for_api.append({"role": "user", "content": final_user_content})

        logger.info(f"LLM 요청 생성 중 (모델: {model}, 대화 턴: {len(history)//2 + 1})")
        # logger.debug(f"Messages sent to OpenAI API (llm_handler.py):\n{messages_for_api}") # 필요시 주석 해제

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages_for_api,
                temperature=0.3, # 일관성 있는 답변 선호 시 낮게 설정
            )
            answer = response.choices[0].message.content.strip()
            logger.info("LLM 답변 생성 완료 (llm_handler.py).")
            return answer
        except Exception as e:
            logger.error(f"OpenAI API 호출 중 오류 발생 (llm_handler.py): {e}")
            # traceback.print_exc() # 개발 중 상세 오류 확인 시 주석 해제
            return None

    # --- 기존 generate_answer 메소드 (선택 사항) ---
    # 만약 이전 버전과의 호환성이나 다른 용도로 필요하다면 남겨둘 수 있습니다.
    # 하이브리드 RAG 시스템에서는 generate_answer_with_history 를 주로 사용합니다.
    def generate_answer(self, question, context_info, model="gpt-3.5-turbo"):
        """
        (레거시 또는 특정 용도) 주어진 질문과 컨텍스트 정보만으로 LLM 답변을 생성합니다. (대화 기록 미사용)
        """
        if not self.client:
             logger.error("OpenAI 클라이언트 미초기화.")
             return None

        system_prompt_legacy = """당신은 제공된 RDF 데이터베이스 정보를 바탕으로 사용자의 질문에 자연스럽게 답변하는 AI 비서입니다.
제공된 '컨텍스트 정보'를 최대한 활용하여 질문에 답해주세요. 정보가 부족하면 솔직하게 부족하다고 답변해주세요.
논문 관련 질문 시 'Google Scholar 등 검색 엔진을 활용해 보세요.' 추가."""

        user_prompt_legacy = f"""[사용자 질문]\n{question}\n\n[컨텍스트 정보]\n{context_info}\n\n[답변]"""

        logger.info(f"LLM 요청 생성 중 (모델: {model}, 레거시 방식)")
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt_legacy},
                    {"role": "user", "content": user_prompt_legacy}
                ],
                temperature=0.7,
            )
            answer = response.choices[0].message.content.strip()
            logger.info("LLM 답변 생성 완료 (레거시 방식).")
            return answer
        except Exception as e:
            logger.error(f"OpenAI API 호출 중 오류 발생 (레거시 방식): {e}")
            return None