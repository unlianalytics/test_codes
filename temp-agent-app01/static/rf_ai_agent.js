// RF AI Agent Chat Functionality
// Version: 1.0
// Last Updated: [Current Date]
// Protected Functions: sendMessage, addMessageToChat, loadKnowledgeStatus

(function() {
    'use strict';
    
    // Register this feature
    if (typeof CodeProtection !== 'undefined') {
        CodeProtection.registerFeature('rf-ai-agent');
    }
    
    document.addEventListener('DOMContentLoaded', function() {
        console.log('[RF AI Agent] Initializing chat functionality...');
        
        // Chat elements
        const chatForm = document.getElementById('chatForm');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const chatMessages = document.getElementById('chatMessages');
        const knowledgeStatus = document.getElementById('knowledgeStatus');
        
        if (!chatForm) {
            console.error('[RF AI Agent] Chat form not found!');
            return;
        }
        
        console.log('[RF AI Agent] Chat elements found, setting up event listeners...');
        
        // Load knowledge base status on page load
        loadKnowledgeStatus();
        
        // Handle form submission
        chatForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            console.log('[RF AI Agent] Form submitted');
            await sendMessage();
        });
        
        // Handle enter key
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                console.log('[RF AI Agent] Enter key pressed');
                sendMessage();
            }
        });
        
        // Send message function - PROTECTED
        window.sendMessage = async function() {
            const message = messageInput.value.trim();
            if (!message) {
                console.log('[RF AI Agent] Empty message, ignoring');
                return;
            }
            
            console.log('[RF AI Agent] Sending message:', message);
            
            // Disable input and button
            messageInput.disabled = true;
            sendButton.disabled = true;
            sendButton.innerHTML = '<span class="button-text">Sending...</span><span class="button-icon">⏳</span>';
            
            // Add user message to chat
            addMessageToChat('user', message);
            
            // Clear input
            messageInput.value = '';
            
            // Add "thinking" message immediately
            const thinkingMessageId = addThinkingMessage();
            
            try {
                // Send to backend
                const formData = new FormData();
                formData.append('message', message);
                
                console.log('[RF AI Agent] Sending request to /chat endpoint...');
                
                const response = await fetch('/chat', {
                    method: 'POST',
                    body: formData
                });
                
                console.log('[RF AI Agent] Response status:', response.status);
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log('[RF AI Agent] Response received:', data);
                
                // Remove thinking message
                removeThinkingMessage(thinkingMessageId);
                
                // Add bot response to chat
                addMessageToChat('bot', data.response);
                
                // Update knowledge status if RAG metrics are available
                if (data.rag_metrics) {
                    updateKnowledgeStatus(data.rag_metrics);
                }
                
            } catch (error) {
                console.error('[RF AI Agent] Error sending message:', error);
                
                // Remove thinking message
                removeThinkingMessage(thinkingMessageId);
                
                // Add error message
                addMessageToChat('bot', 'Sorry, I encountered an error while processing your request. Please try again.');
            } finally {
                // Re-enable input and button
                messageInput.disabled = false;
                sendButton.disabled = false;
                sendButton.innerHTML = '<span class="button-text">Send</span><span class="button-icon">📱</span>';
                messageInput.focus();
            }
        };
        
        // Add message to chat function - PROTECTED
        window.addMessageToChat = function(sender, message) {
            console.log(`[RF AI Agent] Adding ${sender} message to chat`);
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            
            if (sender === 'bot') {
                const avatarDiv = document.createElement('div');
                avatarDiv.className = 'bot-avatar';
                avatarDiv.textContent = '📱';
                contentDiv.appendChild(avatarDiv);
            }
            
            const textDiv = document.createElement('div');
            textDiv.className = 'message-text';
            textDiv.innerHTML = message;
            
            contentDiv.appendChild(textDiv);
            messageDiv.appendChild(contentDiv);
            chatMessages.appendChild(messageDiv);
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            console.log(`[RF AI Agent] Message added successfully`);
        };
        
        // Add thinking message with JavaScript-based animation
        function addThinkingMessage() {
            console.log('[RF AI Agent] Adding thinking message...');
            
            const thinkingDiv = document.createElement('div');
            thinkingDiv.className = 'message bot-message thinking-message';
            thinkingDiv.id = 'thinking-message-' + Date.now();
            
            // Add inline styles to ensure animation works
            thinkingDiv.style.cssText = `
                opacity: 0.9;
                transition: opacity 0.3s ease;
            `;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            
            const avatarDiv = document.createElement('div');
            avatarDiv.className = 'bot-avatar';
            avatarDiv.textContent = '🤔';
            avatarDiv.style.cssText = `
                display: inline-block;
                margin-right: 10px;
                font-size: 24px;
                animation: thinkingPulse 2s infinite;
            `;
            
            const textDiv = document.createElement('div');
            textDiv.className = 'message-text thinking-text';
            textDiv.style.cssText = `
                font-style: italic;
                color: #666;
                display: inline-block;
            `;
            
            // Create the thinking text
            const thinkingText = document.createElement('em');
            thinkingText.textContent = 'Thinking, partner';
            
            // Create animated dots container
            const dotsContainer = document.createElement('span');
            dotsContainer.id = 'dots-' + Date.now();
            dotsContainer.style.cssText = `
                display: inline-block;
                margin-left: 4px;
                font-weight: bold;
                color: #e20074;
            `;
            
            // Create individual dots
            for (let i = 0; i < 3; i++) {
                const dot = document.createElement('span');
                dot.textContent = '.';
                dot.style.cssText = `
                    display: inline-block;
                    animation: dotBounce 1.4s infinite;
                    animation-delay: ${i * 0.2}s;
                    font-size: 18px;
                `;
                dotsContainer.appendChild(dot);
            }
            
            textDiv.appendChild(thinkingText);
            textDiv.appendChild(dotsContainer);
            
            contentDiv.appendChild(avatarDiv);
            contentDiv.appendChild(textDiv);
            thinkingDiv.appendChild(contentDiv);
            chatMessages.appendChild(thinkingDiv);
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            // Add CSS animations dynamically
            addThinkingAnimations();
            
            console.log('[RF AI Agent] Thinking message added with ID:', thinkingDiv.id);
            return thinkingDiv.id;
        }
        
        // Add CSS animations dynamically to ensure they work
        function addThinkingAnimations() {
            // Check if animations are already added
            if (document.getElementById('thinking-animations')) {
                return;
            }
            
            const style = document.createElement('style');
            style.id = 'thinking-animations';
            style.textContent = `
                @keyframes thinkingPulse {
                    0%, 100% { transform: scale(1); }
                    50% { transform: scale(1.1); }
                }
                
                @keyframes dotBounce {
                    0%, 80%, 100% {
                        opacity: 0.3;
                        transform: scale(0.8);
                    }
                    40% {
                        opacity: 1;
                        transform: scale(1.3);
                    }
                }
            `;
            document.head.appendChild(style);
            console.log('[RF AI Agent] Thinking animations added to page');
        }
        
        // Remove thinking message
        function removeThinkingMessage(messageId) {
            console.log('[RF AI Agent] Removing thinking message with ID:', messageId);
            const thinkingMessage = document.getElementById(messageId);
            if (thinkingMessage) {
                thinkingMessage.remove();
                console.log('[RF AI Agent] Thinking message removed successfully');
            } else {
                console.warn('[RF AI Agent] Thinking message not found for removal');
            }
        }
        
        // Load knowledge base status
        async function loadKnowledgeStatus() {
            try {
                console.log('[RF AI Agent] Loading knowledge base status...');
                const response = await fetch('/knowledge-status');
                const data = await response.json();
                
                console.log('[RF AI Agent] Knowledge status response:', data);
                
                if (data.status === 'success') {
                    knowledgeStatus.innerHTML = `
                        <span class="status-icon">📖</span>
                        <span class="status-text">Knowledge base: ${data.knowledge_chunks} chunks loaded</span>
                    `;
                } else {
                    knowledgeStatus.innerHTML = `
                        <span class="status-icon">⚠️</span>
                        <span class="status-text">Knowledge base: ${data.message}</span>
                    `;
                }
            } catch (error) {
                console.error('[RF AI Agent] Error loading knowledge status:', error);
                knowledgeStatus.innerHTML = `
                    <span class="status-icon">❌</span>
                    <span class="status-text">Knowledge base: Unable to load status</span>
                `;
            }
        }
        
        // Update knowledge status with RAG metrics
        function updateKnowledgeStatus(metrics) {
            const docsRetrieved = metrics.documents_retrieved || 0;
            const avgRelevance = metrics.average_relevance || 0;
            const responseTime = metrics.response_time_seconds || 0;
            
            knowledgeStatus.innerHTML = `
                <span class="status-icon">📊</span>
                <span class="status-text">RAG: ${docsRetrieved} docs, ${(avgRelevance * 100).toFixed(1)}% relevance, ${responseTime}s</span>
            `;
        }
        
        console.log('[RF AI Agent] Chat functionality initialized successfully!');
    });
})(); 