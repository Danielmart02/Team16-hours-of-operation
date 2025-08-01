// AI Chat Widget Functionality
class ChatWidget {
    constructor() {
        this.isOpen = false;
        this.isTyping = false;
        this.isFullscreen = false;
        this.isMinimized = false;
        this.agentAvailable = null;
        this.originalSize = { width: 380, height: 500 };
        this.currentSize = { ...this.originalSize };
        
        this.initializeElements();
        this.bindEvents();
        this.checkAgentStatus();
        this.setupResizing();
    }

    initializeElements() {
        this.chatToggle = document.getElementById('chat-toggle');
        this.chatContainer = document.getElementById('chat-container');
        this.chatMessages = document.getElementById('chat-messages');
        this.chatInput = document.getElementById('chat-input');
        this.chatSend = document.getElementById('chat-send');
        this.chatMinimize = document.getElementById('chat-minimize');
        this.chatFullscreen = document.getElementById('chat-fullscreen');
        this.chatClose = document.getElementById('chat-close');
        this.resizeHandle = document.getElementById('resize-handle');
    }

    bindEvents() {
        // Main toggle
        this.chatToggle.addEventListener('click', () => this.toggleChat());
        
        // Header controls
        this.chatMinimize.addEventListener('click', () => this.minimizeChat());
        this.chatFullscreen.addEventListener('click', () => this.toggleFullscreen());
        this.chatClose.addEventListener('click', () => this.closeChat());
        
        // Message sending
        this.chatSend.addEventListener('click', () => this.sendMessage());
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        this.chatInput.addEventListener('input', () => {
            this.chatInput.style.height = 'auto';
            const maxHeight = this.isFullscreen ? 150 : 120;
            this.chatInput.style.height = Math.min(this.chatInput.scrollHeight, maxHeight) + 'px';
        });

        // Close chat when clicking outside (but not in fullscreen)
        document.addEventListener('click', (e) => {
            if (this.isOpen && !this.isFullscreen && 
                !this.chatToggle.contains(e.target) && 
                !this.chatContainer.contains(e.target)) {
                this.closeChat();
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (this.isOpen) {
                if (e.key === 'Escape' && this.isFullscreen) {
                    this.toggleFullscreen();
                } else if (e.key === 'Escape' && !this.isFullscreen) {
                    this.closeChat();
                } else if (e.key === 'F11' && this.isOpen) {
                    e.preventDefault();
                    this.toggleFullscreen();
                }
            }
        });
    }

    setupResizing() {
        let isResizing = false;
        let startX, startY, startWidth, startHeight, startRight, startBottom;

        this.resizeHandle.addEventListener('mousedown', (e) => {
            if (this.isFullscreen) return;
            
            isResizing = true;
            startX = e.clientX;
            startY = e.clientY;
            
            const rect = this.chatContainer.getBoundingClientRect();
            startWidth = rect.width;
            startHeight = rect.height;
            startRight = window.innerWidth - rect.right;
            startBottom = window.innerHeight - rect.bottom;
            
            document.addEventListener('mousemove', handleResize);
            document.addEventListener('mouseup', stopResize);
            
            e.preventDefault();
        });

        const handleResize = (e) => {
            if (!isResizing) return;
            
            const deltaX = e.clientX - startX;
            const deltaY = e.clientY - startY;
            
            // Calculate new dimensions (left-side resizing means width increases when dragging left)
            const newWidth = startWidth - deltaX;
            const newHeight = startHeight + deltaY;
            
            // Apply constraints
            const minWidth = 320;
            const minHeight = 400;
            const maxWidth = window.innerWidth * 0.9;
            const maxHeight = window.innerHeight * 0.9;
            
            const constrainedWidth = Math.max(minWidth, Math.min(maxWidth, newWidth));
            const constrainedHeight = Math.max(minHeight, Math.min(maxHeight, newHeight));
            
            // Update size
            this.chatContainer.style.width = constrainedWidth + 'px';
            this.chatContainer.style.height = constrainedHeight + 'px';
            
            // Adjust position to keep the right edge fixed
            const widthDiff = constrainedWidth - startWidth;
            this.chatContainer.style.right = (startRight - widthDiff) + 'px';
            
            this.currentSize = { width: constrainedWidth, height: constrainedHeight };
        };

        const stopResize = () => {
            isResizing = false;
            document.removeEventListener('mousemove', handleResize);
            document.removeEventListener('mouseup', stopResize);
        };
    }

    async checkAgentStatus() {
        try {
            const response = await fetch('/api/chat/status');
            const status = await response.json();
            this.agentAvailable = status.agent_available;
            
            this.updateAgentStatusUI(status);
        } catch (error) {
            console.error('Failed to check agent status:', error);
            this.agentAvailable = false;
        }
    }

    updateAgentStatusUI(status) {
        const chatInfo = document.querySelector('.chat-info p');
        if (status.agent_available) {
            chatInfo.textContent = 'Ask me about staffing predictions';
            this.chatToggle.style.background = 'linear-gradient(135deg, var(--cpp-green) 0%, var(--cpp-accent) 100%)';
        } else {
            chatInfo.textContent = 'Limited responses available';
            this.chatToggle.style.background = 'linear-gradient(135deg, #666 0%, #888 100%)';
        }
    }

    toggleChat() {
        if (this.isOpen) {
            this.closeChat();
        } else {
            this.openChat();
        }
    }

    openChat() {
        this.isOpen = true;
        this.isMinimized = false;
        this.chatContainer.classList.add('active');
        this.chatContainer.classList.remove('minimized');
        this.updateToggleButton();
        
        // Restore size if not fullscreen
        if (!this.isFullscreen) {
            this.chatContainer.style.width = this.currentSize.width + 'px';
            this.chatContainer.style.height = this.currentSize.height + 'px';
            // Reset right positioning to default
            this.chatContainer.style.right = '0px';
        }
        
        setTimeout(() => this.chatInput.focus(), 300);
    }

    closeChat() {
        this.isOpen = false;
        this.isMinimized = false;
        this.chatContainer.classList.remove('active', 'minimized');
        
        // Exit fullscreen if active
        if (this.isFullscreen) {
            this.exitFullscreen();
        }
        
        this.updateToggleButton();
    }

    minimizeChat() {
        if (this.isFullscreen) {
            this.exitFullscreen();
        }
        
        this.isMinimized = !this.isMinimized;
        
        if (this.isMinimized) {
            this.chatContainer.style.height = '60px';
            this.chatContainer.classList.add('minimized');
        } else {
            this.chatContainer.style.height = this.currentSize.height + 'px';
            this.chatContainer.classList.remove('minimized');
        }
    }

    toggleFullscreen() {
        if (this.isFullscreen) {
            this.exitFullscreen();
        } else {
            this.enterFullscreen();
        }
    }

    enterFullscreen() {
        this.isFullscreen = true;
        this.isMinimized = false;
        
        // Store current size before going fullscreen
        if (!this.chatContainer.classList.contains('fullscreen')) {
            const rect = this.chatContainer.getBoundingClientRect();
            this.currentSize = { width: rect.width, height: rect.height };
        }
        
        this.chatContainer.classList.add('fullscreen');
        this.chatContainer.classList.remove('minimized');
        this.chatFullscreen.innerHTML = '🗗'; // Restore icon
        this.chatFullscreen.title = 'Exit Fullscreen';
        
        // Focus input
        setTimeout(() => this.chatInput.focus(), 100);
    }

    exitFullscreen() {
        this.isFullscreen = false;
        this.chatContainer.classList.remove('fullscreen');
        this.chatFullscreen.innerHTML = '⛶'; // Fullscreen icon
        this.chatFullscreen.title = 'Toggle Fullscreen';
        
        // Restore previous size
        this.chatContainer.style.width = this.currentSize.width + 'px';
        this.chatContainer.style.height = this.currentSize.height + 'px';
    }

    updateToggleButton() {
        if (this.isOpen) {
            this.chatToggle.innerHTML = '💬';
            this.chatToggle.title = 'Chat is open';
        } else {
            this.chatToggle.innerHTML = '🤖';
            this.chatToggle.title = 'Chat with AI Assistant';
        }
    }

    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message || this.isTyping) return;

        // Add user message
        this.addMessage(message, 'user');
        this.chatInput.value = '';
        this.chatInput.style.height = 'auto';

        // Show typing indicator
        this.showTypingIndicator();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            this.hideTypingIndicator();
            
            if (data.error) {
                this.addMessage('Sorry, I encountered an error: ' + data.error, 'bot');
            } else {
                this.addMessage(data.response, 'bot');
                
                if (data.agent_available !== undefined) {
                    this.agentAvailable = data.agent_available;
                }
                
                if (data.fallback) {
                    this.addSystemMessage('Note: AI agent is currently unavailable. Using basic responses.');
                }
            }
        } catch (error) {
            console.error('Chat error:', error);
            this.hideTypingIndicator();
            this.addMessage('Sorry, I\'m having trouble connecting right now. Please try again later.', 'bot');
        }
    }

    addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        messageDiv.textContent = text;
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    addSystemMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message system';
        messageDiv.textContent = text;
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    showTypingIndicator() {
        this.isTyping = true;
        this.chatSend.disabled = true;
        
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        
        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.isTyping = false;
        this.chatSend.disabled = false;
        
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    // Method to clear chat history
    async clearHistory() {
        try {
            const response = await fetch('/api/chat/clear', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (response.ok) {
                const messages = this.chatMessages.querySelectorAll('.message:not(.bot):not(:first-child)');
                messages.forEach(msg => msg.remove());
                
                this.addSystemMessage('Chat history cleared.');
            }
        } catch (error) {
            console.error('Failed to clear chat history:', error);
        }
    }

    // Method to refresh agent status
    async refreshStatus() {
        await this.checkAgentStatus();
    }

    // Method to reset chat size and position
    resetSize() {
        if (!this.isFullscreen) {
            this.currentSize = { ...this.originalSize };
            this.chatContainer.style.width = this.currentSize.width + 'px';
            this.chatContainer.style.height = this.currentSize.height + 'px';
            this.chatContainer.style.right = '0px'; // Reset to default position
        }
    }
}

// Initialize chat widget when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatWidget = new ChatWidget();
    
    // Optional: Add keyboard shortcut info to welcome message
    const welcomeMessage = document.querySelector('.message.bot');
    if (welcomeMessage) {
        welcomeMessage.innerHTML += '<br><br><small><em>💡 Tips: Press F11 for fullscreen, Escape to close, or drag the corner to resize!</em></small>';
    }
});
