/**
 * OpenClaw - NexusMind Web UI
 * Client-side application
 */

class OpenClawApp {
    constructor() {
        this.sessionId = null;
        this.ws = null;
        this.isConnected = false;
        this.init();
    }

    init() {
        this.bindNavigation();
        this.bindChat();
        this.bindMemory();
        this.connectWebSocket();
        this.loadSkills();
        this.loadConfig();
        this.autoResizeTextarea();
    }

    // ── Navigation ──────────────────────────────────────────

    bindNavigation() {
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const view = btn.dataset.view;
                document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById(`view-${view}`).classList.add('active');
            });
        });
    }

    // ── Chat ────────────────────────────────────────────────

    bindChat() {
        const form = document.getElementById('chat-form');
        const input = document.getElementById('chat-input');

        form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
    }

    async sendMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        if (!message) return;

        input.value = '';
        input.style.height = 'auto';
        this.addMessage('user', message);
        this.showTyping();

        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'message', content: message }));
        } else {
            try {
                const resp = await fetch('/api/chat/simple', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message, session_id: this.sessionId }),
                });
                const data = await resp.json();
                this.hideTyping();
                this.sessionId = data.session_id;
                this.addMessage('assistant', data.reply || data.detail || 'Erreur de communication.');
            } catch (err) {
                this.hideTyping();
                this.addMessage('system', `Erreur: ${err.message}`);
            }
        }
    }

    addMessage(role, content) {
        const container = document.getElementById('chat-messages');
        const div = document.createElement('div');
        div.className = `message ${role}`;
        div.innerHTML = `
            <div class="message-content">${this.formatContent(content)}</div>
            <div class="message-meta">${new Date().toLocaleTimeString('fr-FR')}</div>
        `;
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }

    formatContent(text) {
        // Basic markdown-like formatting
        let html = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // Code blocks
        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        // Bold
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Italic
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
        // Line breaks
        html = html.replace(/\n/g, '<br>');

        return html;
    }

    showTyping() {
        const container = document.getElementById('chat-messages');
        const existing = document.getElementById('typing-msg');
        if (existing) return;

        const div = document.createElement('div');
        div.id = 'typing-msg';
        div.className = 'message assistant';
        div.innerHTML = `
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }

    hideTyping() {
        const typing = document.getElementById('typing-msg');
        if (typing) typing.remove();
    }

    // ── WebSocket ───────────────────────────────────────────

    connectWebSocket() {
        const clientId = 'web_' + Math.random().toString(36).substr(2, 9);
        const wsUrl = `ws://${window.location.hostname}:${window.location.port}/ws/${clientId}`;

        try {
            this.ws = new WebSocket(wsUrl);
            this.ws.onopen = () => {
                this.isConnected = true;
                this.updateStatus(true);
            };

            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'start') {
                    this.sessionId = data.session_id;
                } else if (data.type === 'chunk') {
                    this.appendToLastMessage(data.content);
                } else if (data.type === 'end') {
                    this.hideTyping();
                    // Final message already appended via chunks
                }
            };

            this.ws.onclose = () => {
                this.isConnected = false;
                this.updateStatus(false);
                setTimeout(() => this.connectWebSocket(), 3000);
            };

            this.ws.onerror = () => {
                this.isConnected = false;
                this.updateStatus(false);
            };
        } catch (e) {
            this.updateStatus(false);
        }
    }

    appendToLastMessage(content) {
        const messages = document.querySelectorAll('.message.assistant');
        const typing = document.getElementById('typing-msg');

        if (typing) {
            typing.remove();
            this.addMessage('assistant', content);
        } else if (messages.length > 0) {
            const last = messages[messages.length - 1];
            const contentDiv = last.querySelector('.message-content');
            contentDiv.innerHTML += this.formatContent(content);
        }
    }

    updateStatus(online) {
        const indicator = document.getElementById('status-indicator');
        if (online) {
            indicator.className = 'status online';
            indicator.innerHTML = '<span class="dot"></span> Connecte';
        } else {
            indicator.className = 'status';
            indicator.innerHTML = '<span class="dot"></span> Deconnecte';
        }
    }

    // ── Memory ──────────────────────────────────────────────

    bindMemory() {
        const btn = document.getElementById('memory-search-btn');
        const input = document.getElementById('memory-query');
        if (btn) {
            btn.addEventListener('click', () => this.searchMemory(input.value));
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') this.searchMemory(input.value);
            });
        }
        this.loadMemoryCategories();
    }

    async loadMemoryCategories() {
        try {
            const resp = await fetch('/api/memory/categories');
            const data = await resp.json();
            const container = document.getElementById('memory-categories');
            if (container && data.categories) {
                container.innerHTML = data.categories.map(cat => `
                    <div class="card">
                        <h3>${cat.name}</h3>
                        <p>${cat.description || ''} (${cat.item_count} elements)</p>
                    </div>
                `).join('');
            }
        } catch (e) {
            // Memory API not available
        }
    }

    async searchMemory(query) {
        if (!query) return;
        try {
            const resp = await fetch(`/api/memory/search?query=${encodeURIComponent(query)}&top_k=10`);
            const data = await resp.json();
            const container = document.getElementById('memory-results');
            if (container && data.results) {
                container.innerHTML = `<h3>Resultats pour "${query}"</h3>` +
                    data.results.map(r => `
                        <div class="card">
                            <h3>${r.category || 'general'}</h3>
                            <p>${r.content || ''}</p>
                        </div>
                    `).join('');
            }
        } catch (e) {
            console.error('Memory search error:', e);
        }
    }

    // ── Skills ──────────────────────────────────────────────

    async loadSkills() {
        try {
            const resp = await fetch('/api/skills');
            const data = await resp.json();
            const container = document.getElementById('skills-list');
            if (container && data.skills) {
                container.innerHTML = data.skills.map(s => `
                    <div class="card">
                        <h3>${s.name} <span style="color:var(--text-muted)">v${s.version}</span></h3>
                        <p>${s.description}</p>
                        <p style="margin-top:8px;color:var(--text-muted)">${(s.tags || []).map(t => '#'+t).join(' ')}</p>
                    </div>
                `).join('');
            }
        } catch (e) {
            // Skills API not available
        }
    }

    // ── Config ──────────────────────────────────────────────

    async loadConfig() {
        try {
            const resp = await fetch('/api/config');
            const data = await resp.json();
            const container = document.getElementById('config-editor');
            if (container) {
                container.innerHTML = `<pre style="background:var(--bg-tertiary);padding:16px;border-radius:8px;overflow:auto;max-height:calc(100vh - 150px)">${JSON.stringify(data, null, 2)}</pre>`;
            }
        } catch (e) {
            // Config API not available
        }
    }

    // ── Utilities ───────────────────────────────────────────

    autoResizeTextarea() {
        const textarea = document.getElementById('chat-input');
        if (textarea) {
            textarea.addEventListener('input', () => {
                textarea.style.height = 'auto';
                textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
            });
        }
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    window.app = new OpenClawApp();
});
