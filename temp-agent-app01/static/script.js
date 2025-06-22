// Customer Support Modal Functions - Move these to global scope
function openSupportModal() {
    console.log('openSupportModal called');
    const modal = document.getElementById('supportModal');
    console.log('Modal element:', modal);
    if (modal) {
        modal.style.display = 'block';
        document.body.style.overflow = 'hidden'; // Prevent background scrolling
        console.log('Modal should be visible now');
    } else {
        console.error('Modal element not found!');
    }
}

function closeSupportModal() {
    console.log('closeSupportModal called');
    const modal = document.getElementById('supportModal');
    if (modal) {
        modal.style.display = 'none';
        document.body.style.overflow = 'auto'; // Restore scrolling
        console.log('Modal should be hidden now');
    } else {
        console.error('Modal element not found!');
    }
}

// Close modal when clicking outside of it
function handleModalClick(event) {
    const modal = document.getElementById('supportModal');
    if (event.target === modal) {
        closeSupportModal();
    }
}

// Close modal with Escape key
function handleKeyDown(event) {
    if (event.key === 'Escape') {
        closeSupportModal();
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chatForm');
    const messageInput = document.getElementById('messageInput');
    const chatMessages = document.getElementById('chatMessages');
    const sendButton = document.getElementById('sendButton');
    const knowledgeStatus = document.getElementById('knowledgeStatus');

    // Check knowledge base status on page load
    checkKnowledgeStatus();

    // Handle form submission
    chatForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const message = messageInput.value.trim();
        if (!message) return;

        // Add user message to chat
        addMessage(message, 'user');
        messageInput.value = '';

        // Show loading message
        const loadingId = addLoadingMessage();

        try {
            // Send message to backend
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `message=${encodeURIComponent(message)}`
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Remove loading message
            removeLoadingMessage(loadingId);
            
            // Add bot response
            addMessage(data.response, 'bot');

        } catch (error) {
            console.error('Error:', error);
            removeLoadingMessage(loadingId);
            addMessage('Sorry partner, I\'m having some technical difficulties. Please try again!', 'bot');
        }
    });

    // Check knowledge base status
    async function checkKnowledgeStatus() {
        try {
            const response = await fetch('/knowledge-status');
            const data = await response.json();
            
            if (data.status === 'success') {
                knowledgeStatus.innerHTML = `
                    <span class="status-icon">📚</span>
                    <span class="status-text">Knowledge base loaded: ${data.knowledge_chunks} chunks</span>
                `;
                knowledgeStatus.className = 'knowledge-status loaded';
            } else {
                knowledgeStatus.innerHTML = `
                    <span class="status-icon">⚠️</span>
                    <span class="status-text">Knowledge base error</span>
                `;
                knowledgeStatus.className = 'knowledge-status error';
            }
        } catch (error) {
            console.error('Error checking knowledge status:', error);
            knowledgeStatus.innerHTML = `
                <span class="status-icon">❌</span>
                <span class="status-text">Cannot connect to knowledge base</span>
            `;
            knowledgeStatus.className = 'knowledge-status error';
        }
    }

    // Add message to chat
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const avatar = sender === 'bot' ? '🤠' : '👤';
        const avatarClass = sender === 'bot' ? 'bot-avatar' : 'user-avatar';
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="${avatarClass}">${avatar}</div>
                <div class="message-text">${escapeHtml(text)}</div>
            </div>
        `;
        
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    // Add loading message
    function addLoadingMessage() {
        const loadingId = 'loading-' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        messageDiv.id = loadingId;
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="bot-avatar">🤠</div>
                <div class="message-text">
                    <div class="loading">
                        Thinking, partner...
                        <div class="loading-dots">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
        return loadingId;
    }

    // Remove loading message
    function removeLoadingMessage(loadingId) {
        const loadingElement = document.getElementById(loadingId);
        if (loadingElement) {
            loadingElement.remove();
        }
    }

    // Scroll to bottom of chat
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Escape HTML to prevent XSS
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Handle Enter key in input
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    // Auto-focus input on page load
    messageInput.focus();

    // Add some Texas-themed welcome messages with RAG awareness
    const texasGreetings = [
        "Howdy partner! I'm your Texas AI assistant with enhanced knowledge capabilities. What can I help you with today?",
        "Welcome to Texas AI! I've got access to some deep knowledge that might help with your questions. What's on your mind?",
        "Well hello there! I'm ready to chat about anything, and I've got some special knowledge up my sleeve too!",
        "Howdy! I'm here to help with whatever you need, and I've got some enhanced knowledge to make our conversation even better!"
    ];

    // Change the initial greeting randomly
    const randomGreeting = texasGreetings[Math.floor(Math.random() * texasGreetings.length)];
    const initialMessage = document.querySelector('.bot-message .message-text');
    if (initialMessage) {
        initialMessage.textContent = randomGreeting;
    }

    // Add some fun Texas-themed interactions
    let messageCount = 0;
    
    chatForm.addEventListener('submit', function() {
        messageCount++;
        
        // Add some Texas flair after a few messages
        if (messageCount === 3) {
            setTimeout(() => {
                addMessage("Y'all are really getting into this chat! And I'm using my enhanced knowledge to give you the best answers possible! 🤠", 'bot');
            }, 1000);
        }
    });

    // Add event listeners for modal functionality
    window.addEventListener('click', handleModalClick);
    document.addEventListener('keydown', handleKeyDown);
}); 