document.addEventListener('DOMContentLoaded', function() {
    const messageContainer = document.getElementById('messageContainer');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const templates = document.querySelectorAll('.template-card');

    // Template click handlers
    templates.forEach(template => {
        template.addEventListener('click', function() {
            const question = this.querySelector('h3').textContent;
            userInput.value = question;
            sendMessage();
        });
    });

    // Send message function
    function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Add user message
        appendMessage(message, true);
        userInput.value = '';

        // Show typing indicator
        const typingIndicator = appendTypingIndicator();

        // Send to backend
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: message })
        })
        .then(response => response.json())
        .then(data => {
            // Remove typing indicator
            typingIndicator.remove();

            if (data.success) {
                appendMessage(data.answer, false);
            } else {
                appendMessage("Sorry, I encountered an error. Please try again.", false);
            }
        })
        .catch(error => {
            typingIndicator.remove();
            console.error('Error:', error);
            appendMessage("Sorry, I encountered an error. Please try again.", false);
        });
    }

    // Append message to chat
    function appendMessage(content, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.className = isUser ? 'message-user' : 'message-bot';
        messageDiv.textContent = content;

        const wrapper = document.createElement('div');
        wrapper.className = `flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`;
        wrapper.appendChild(messageDiv);

        messageContainer.appendChild(wrapper);
        messageContainer.scrollTop = messageContainer.scrollHeight;
    }

    // Add typing indicator
    function appendTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'message-bot loading-dots';
        indicator.textContent = "Thinking";

        const wrapper = document.createElement('div');
        wrapper.className = 'flex justify-start mb-4';
        wrapper.appendChild(indicator);

        messageContainer.appendChild(wrapper);
        messageContainer.scrollTop = messageContainer.scrollHeight;
        return wrapper;
    }

    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});