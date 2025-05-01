const chatbox = document.getElementById('chatbox');
const messageForm = document.getElementById('messageForm');
const userInput = document.getElementById('userInput');
const suggestionsContainer = document.getElementById('suggestions');

// 메시지를 채팅창에 추가하는 함수
function addMessage(message, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender + '-message');
    messageDiv.textContent = message;
    chatbox.appendChild(messageDiv);
    // 새 메시지가 추가되면 스크롤을 맨 아래로 이동
    chatbox.scrollTop = chatbox.scrollHeight;
}

// 폼 제출 처리
messageForm.addEventListener('submit', async (event) => {
    event.preventDefault(); // 페이지 새로고침 방지
    const userMessage = userInput.value.trim();

    if (userMessage) {
        addMessage(userMessage, 'user'); // 사용자 메시지 표시
        userInput.value = ''; // 입력창 비우기
        
        // 로딩 메시지 (선택 사항)
        addMessage("답변 생성 중...", 'bot');
        const loadingMessageElement = chatbox.lastElementChild; // 로딩 메시지 요소 저장
        
        try {
            // 백엔드로 메시지 전송
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: userMessage }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // 로딩 메시지 제거
             chatbox.removeChild(loadingMessageElement);

            addMessage(data.response, 'bot'); // 봇 응답 표시
        } catch (error) {
             // 로딩 메시지 제거 후 에러 메시지 표시
             if (loadingMessageElement && chatbox.contains(loadingMessageElement)) {
                 chatbox.removeChild(loadingMessageElement);
             }
            console.error('Error sending message:', error);
            addMessage('죄송합니다. 메시지 처리 중 오류가 발생했습니다.', 'bot');
        }
    }
});

// 추천 질문 버튼 처리
suggestionsContainer.addEventListener('click', (event) => {
    if (event.target.classList.contains('suggestion-btn')) {
        const question = event.target.textContent;
        userInput.value = question; // 입력창에 질문 채우기
        // 선택적으로 바로 전송
        messageForm.requestSubmit(); // 폼 제출 이벤트 발생시키기
        userInput.focus(); // 입력창에 포커스 주기
    }
});