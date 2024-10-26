// Global variables
let chatHistory = [];
let currentConversationId = null;

// Initialize the chat interface
document.addEventListener('DOMContentLoaded', function () {
    loadChatHistory();
    initializeChatForm();

    // Add event listener for new chat button
    document.querySelector('[onclick="newChat()"]').addEventListener('click', newChat);
});

// Initialize chat form
function initializeChatForm() {
    const form = document.getElementById('chatForm');
    const input = document.getElementById('userInput');
    const submitButton = form.querySelector('button[type="submit"]');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = input.value.trim();

        if (message) {
            // Disable input and button while processing
            input.disabled = true;
            submitButton.disabled = true;

            // Hide templates when starting a chat
            const templates = document.getElementById('initialTemplates');
            if (templates) {
                templates.style.display = 'none';
            }

            try {
                await sendMessage(message);
            } catch (error) {
                console.error('Error sending message:', error);
                appendMessage('Sorry, there was an error sending your message. Please try again.', 'error');
            } finally {
                // Re-enable input and button
                input.disabled = false;
                submitButton.disabled = false;
                input.value = '';
                input.focus();
            }
        }
    });
}

// Send message to the server
async function sendMessage(message) {
    // Add user message to chat with typing animation
    appendMessage(message, 'user');

    // Show loading indicator
    showLoading();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: message,
                chat_history: chatHistory,
                conversation_id: currentConversationId
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Remove loading indicator
        hideLoading();

        if (data.error) {
            appendMessage('Sorry, there was an error processing your request.', 'error');
        } else {
            // Add bot response with typing animation
            appendMessage(data.response, 'bot');

            // Update chat history
            chatHistory.push({
                question: message,
                answer: data.response
            });

            // Update conversation ID and reload chat history
            if (data.conversation_id) {
                currentConversationId = data.conversation_id;
                await loadChatHistory();
            }
        }
    } catch (error) {
        hideLoading();
        appendMessage('Sorry, there was an error connecting to the server.', 'error');
        console.error('Error:', error);
    }
}

// Load chat history with error handling and retry mechanism
async function loadChatHistory(retryCount = 3) {
    const chatHistoryDiv = document.getElementById('chatHistory');

    for (let i = 0; i < retryCount; i++) {
        try {
            const response = await fetch('/chat_history');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const history = await response.json();

            // Clear existing history
            chatHistoryDiv.innerHTML = '';

            // Add history items with fade-in animation
            history.forEach((chat, index) => {
                const chatDiv = document.createElement('div');
                chatDiv.className = 'chat-history-item p-2 hover:bg-gray-700 rounded cursor-pointer opacity-0 transition-opacity duration-300';
                chatDiv.style.animationDelay = `${index * 50}ms`;

                chatDiv.innerHTML = `
                    <div class="font-medium truncate">${escapeHtml(chat.title)}</div>
                    <div class="text-sm text-gray-400 truncate">${escapeHtml(chat.preview)}</div>
                    <div class="text-xs text-gray-500">${formatTimestamp(chat.timestamp)}</div>
                `;

                chatDiv.addEventListener('click', () => loadConversation(chat.id));

                chatHistoryDiv.appendChild(chatDiv);

                // Trigger fade-in
                setTimeout(() => {
                    chatDiv.classList.add('opacity-100');
                }, 50);
            });

            return; // Success, exit retry loop
        } catch (error) {
            console.error(`Error loading chat history (attempt ${i + 1}):`, error);
            if (i === retryCount - 1) {
                // Show error message on final retry
                chatHistoryDiv.innerHTML = `
                    <div class="text-red-500 p-2">
                        Failed to load chat history. Please refresh the page.
                    </div>`;
            }
        }
    }
}

// Load a specific conversation
async function loadConversation(conversationId) {
    try {
        const response = await fetch(`/conversation/${conversationId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const conversation = await response.json();

        // Clear current chat
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.innerHTML = '';

        // Update global variables
        currentConversationId = conversationId;
        chatHistory = [];

        // Hide templates
        const templates = document.getElementById('initialTemplates');
        if (templates) {
            templates.style.display = 'none';
        }

        // Display conversation messages with typing animation
        for (const msg of conversation.messages) {
            await appendMessage(msg.question, 'user');
            await appendMessage(msg.answer, 'bot');
            chatHistory.push(msg);
        }

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    } catch (error) {
        console.error('Error loading conversation:', error);
        appendMessage('Failed to load conversation. Please try again.', 'error');
    }
}

// Start new chat with animation
function newChat() {
    const chatMessages = document.getElementById('chatMessages');

    // Fade out existing messages
    chatMessages.style.opacity = '0';

    setTimeout(() => {
        // Reset state
        chatHistory = [];
        currentConversationId = null;
        chatMessages.innerHTML = '';
        document.getElementById('userInput').value = '';

        // Show templates
        const templates = document.getElementById('initialTemplates');
        if (templates) {
            templates.style.display = 'flex';
        }

        // Fade in new content
        chatMessages.style.opacity = '1';
    }, 300);
}

// Load template question
function loadTemplate(template) {
    const input = document.getElementById('userInput');
    input.value = template;
    input.focus();
    document.getElementById('chatForm').dispatchEvent(new Event('submit'));
}

// Append message with typing animation
async function appendMessage(message, type) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message opacity-0 transition-opacity duration-300`;

    // Add appropriate styling based on message type
    switch (type) {
        case 'user':
            messageDiv.classList.add('bg-blue-100', 'text-blue-900', 'rounded-lg', 'p-3', 'mb-2', 'max-w-3/4', 'ml-auto');
            break;
        case 'bot':
            messageDiv.classList.add('bg-gray-100', 'text-gray-900', 'rounded-lg', 'p-3', 'mb-2', 'max-w-3/4');
            break;
        case 'error':
            messageDiv.classList.add('bg-red-100', 'text-red-900', 'rounded-lg', 'p-3', 'mb-2', 'max-w-3/4');
            break;
    }

    // Add message content
    messageDiv.innerHTML = escapeHtml(message);

    // Add to chat
    chatMessages.appendChild(messageDiv);

    // Scroll to new message
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Trigger fade-in
    await new Promise(resolve => setTimeout(resolve, 50));
    messageDiv.classList.add('opacity-100');

    // Wait for animation to complete
    await new Promise(resolve => setTimeout(resolve, 300));
}

// Show loading indicator
function showLoading() {
    const chatMessages = document.getElementById('chatMessages');
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading-indicator flex items-center space-x-2 p-3 mb-2';
    loadingDiv.innerHTML = `
        <div class="loading-dots flex space-x-1">
            <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
            <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
            <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.4s"></div>
        </div>
    `;
    loadingDiv.id = 'loadingIndicator';
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Hide loading indicator
function hideLoading() {
    const loadingIndicator = document.getElementById('loadingIndicator');
    if (loadingIndicator) {
        loadingIndicator.remove();
    }
}

// Format timestamp
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`;
}

// Escape HTML to prevent XSS attacks
function escapeHtml(html) {
    const div = document.createElement('div');
    div.textContent = html;
    return div.innerHTML;
}
