document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    function addMessage(content, sender) {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message', sender);
        
        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        
        // Simple text formatting for newlines
        contentDiv.innerHTML = content.replace(/\n/g, '<br>');
        
        msgDiv.appendChild(contentDiv);
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
        return msgDiv;
    }

    function addTypingIndicator() {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message', 'ai');
        msgDiv.id = 'typing-indicator-msg';
        
        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        
        contentDiv.innerHTML = `
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        
        msgDiv.appendChild(contentDiv);
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
    }

    function removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator-msg');
        if (indicator) {
            indicator.remove();
        }
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function handleSend() {
        const text = userInput.value.trim();
        if (!text) return;

        // Add user message
        addMessage(text, 'user');
        userInput.value = '';
        
        // Add typing indicator
        addTypingIndicator();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: text })
            });

            removeTypingIndicator();

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            addMessage(data.response, 'ai');
            
        } catch (error) {
            removeTypingIndicator();
            addMessage("Sorry, I encountered an error connecting to the server.", 'ai');
            console.error('Error:', error);
        }
    }

    sendBtn.addEventListener('click', handleSend);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSend();
        }
    });
    
    // Focus input on load
    userInput.focus();
});
