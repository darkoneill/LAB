/**
 * OpenClaw - NexusMind Web UI
 * Client-side application
 */

class OpenClawApp {
    constructor() {
        this.sessionId = null;
        this.ws = null;
        this.isConnected = false;
        this.wsStreamBuffer = '';
        this.wsStreamTarget = null;
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
        this.checkSystemHealth();
    }

    // ── System Health Check ──────────────────────────────────

    async checkSystemHealth() {
        try {
            const resp = await fetch('/api/config');
            if (!resp.ok) {
                this.showSetupBanner('Impossible de joindre le serveur. Verifiez les logs.');
                return;
            }
            const config = await resp.json();

            // Check if any LLM provider has an API key configured
            const providers = config.providers || {};
            let hasProvider = false;
            for (const [name, prov] of Object.entries(providers)) {
                if (prov && prov.enabled) {
                    const key = prov.api_key || '';
                    // Keys are redacted to ***xxxx by the API, check for real key
                    if (key && key.length > 6 && !key.startsWith('***')) {
                        hasProvider = true;
                        break;
                    }
                    // Ollama doesn't need API key
                    if (name === 'ollama' && prov.base_url) {
                        hasProvider = true;
                        break;
                    }
                    // Redacted key means it IS configured
                    if (key && key.startsWith('***')) {
                        hasProvider = true;
                        break;
                    }
                }
            }

            if (!hasProvider) {
                this.showSetupBanner(
                    'Aucun fournisseur LLM configure. ' +
                    'Allez dans l\'onglet Config pour ajouter votre cle API (Anthropic, OpenAI, ou Ollama).'
                );
            }
        } catch (e) {
            this.showSetupBanner('Serveur injoignable. Verifiez que le container est demarre.');
        }
    }

    showSetupBanner(message) {
        const existing = document.getElementById('setup-banner');
        if (existing) existing.remove();

        const banner = document.createElement('div');
        banner.id = 'setup-banner';
        banner.className = 'setup-banner';
        banner.innerHTML = `
            <div class="setup-banner-content">
                <span class="setup-icon">&#9888;</span>
                <span class="setup-text">${message}</span>
                <button class="btn-sm setup-goto-config" onclick="document.querySelector('[data-view=config]').click()">
                    Configurer
                </button>
                <button class="btn-sm setup-dismiss" onclick="this.parentElement.parentElement.remove()">
                    &#10005;
                </button>
            </div>
        `;
        const content = document.getElementById('content');
        if (content) content.prepend(banner);
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
                this.hideTyping();

                if (resp.status === 429) {
                    this.addMessage('system', 'Rate limit atteint. Attendez un moment avant de reessayer.');
                    return;
                }
                if (resp.status === 503) {
                    this.addMessage('system',
                        'Le cerveau de l\'agent n\'est pas initialise. ' +
                        'Verifiez qu\'une cle API est configuree dans l\'onglet Config.'
                    );
                    return;
                }
                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({}));
                    this.addMessage('system', 'Erreur serveur: ' + (err.detail || err.error || resp.statusText));
                    return;
                }

                const data = await resp.json();
                this.sessionId = data.session_id;
                this.addMessage('assistant', data.reply || data.content || 'Reponse vide.');
            } catch (err) {
                this.hideTyping();
                this.addMessage('system',
                    'Impossible de joindre le serveur. Verifiez que le service est demarre.\n' +
                    'Erreur: ' + err.message
                );
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
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${clientId}`;

        try {
            this.ws = new WebSocket(wsUrl);
            // Expose for mission-control.js
            window.ws = this.ws;

            this.ws.onopen = () => {
                this.isConnected = true;
                this.updateStatus(true);
            };

            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'start') {
                    this.sessionId = data.session_id;
                    this.hideTyping();
                    this.wsStreamBuffer = '';
                    this.wsStreamTarget = null;
                } else if (data.type === 'chunk') {
                    if (!this.wsStreamTarget) {
                        // Create a new message div for streaming
                        const container = document.getElementById('chat-messages');
                        const div = document.createElement('div');
                        div.className = 'message assistant';
                        div.innerHTML = `
                            <div class="message-content"></div>
                            <div class="message-meta">${new Date().toLocaleTimeString('fr-FR')}</div>
                        `;
                        container.appendChild(div);
                        this.wsStreamTarget = div.querySelector('.message-content');
                    }
                    this.wsStreamBuffer += data.content;
                    this.wsStreamTarget.innerHTML = this.formatContent(this.wsStreamBuffer);
                    const container = document.getElementById('chat-messages');
                    container.scrollTop = container.scrollHeight;
                } else if (data.type === 'end') {
                    this.hideTyping();
                    this.wsStreamTarget = null;
                    this.wsStreamBuffer = '';
                }

                // Forward to mission-control.js listener
                if (window.wsOnMessage) {
                    window.wsOnMessage(event);
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
        const container = document.getElementById('memory-categories');
        if (!container) return;

        try {
            const resp = await fetch('/api/memory/categories');
            if (resp.status === 503) {
                container.innerHTML = '<div class="empty-state">Systeme de memoire non initialise. Verifiez la configuration.</div>';
                return;
            }
            const data = await resp.json();
            if (data.categories && data.categories.length > 0) {
                container.innerHTML = '<h3>Categories</h3>' + data.categories.map(cat => `
                    <div class="card">
                        <h3>${this.esc(cat.name || '')}</h3>
                        <p>${this.esc(cat.description || '')} (${cat.item_count || 0} elements)</p>
                    </div>
                `).join('');
            } else {
                container.innerHTML = '<div class="empty-state">Aucune categorie en memoire. Les souvenirs apparaitront ici au fil des conversations.</div>';
            }
        } catch (e) {
            container.innerHTML = '<div class="empty-state">Memoire indisponible. Le serveur est-il demarre ?</div>';
        }
    }

    async searchMemory(query) {
        if (!query) return;
        const container = document.getElementById('memory-results');
        if (!container) return;

        container.innerHTML = '<div class="empty-state">Recherche en cours...</div>';

        try {
            const resp = await fetch(`/api/memory/search?query=${encodeURIComponent(query)}&top_k=10`);
            if (!resp.ok) {
                container.innerHTML = '<div class="empty-state">Erreur lors de la recherche.</div>';
                return;
            }
            const data = await resp.json();
            if (data.results && data.results.length > 0) {
                container.innerHTML = `<h3>Resultats pour "${this.esc(query)}"</h3>` +
                    data.results.map(r => `
                        <div class="card">
                            <h3>${this.esc(r.category || 'general')}</h3>
                            <p>${this.esc(r.content || '')}</p>
                            ${r.score ? `<div class="card-meta">Score: ${Math.round(r.score * 100)}%</div>` : ''}
                        </div>
                    `).join('');
            } else {
                container.innerHTML = `<div class="empty-state">Aucun resultat pour "${this.esc(query)}".</div>`;
            }
        } catch (e) {
            container.innerHTML = '<div class="empty-state">Erreur reseau. Verifiez la connexion.</div>';
        }
    }

    // ── Skills ──────────────────────────────────────────────

    async loadSkills() {
        const container = document.getElementById('skills-list');
        if (!container) return;

        try {
            const resp = await fetch('/api/skills');
            const data = await resp.json();
            if (data.skills && data.skills.length > 0) {
                container.innerHTML = data.skills.map(s => `
                    <div class="card">
                        <h3>${this.esc(s.name || '')} ${s.version ? `<span style="color:var(--text-muted)">v${this.esc(s.version)}</span>` : ''}</h3>
                        <p>${this.esc(s.description || '')}</p>
                        ${(s.tags && s.tags.length) ? `<p class="card-meta">${s.tags.map(t => '#' + this.esc(t)).join(' ')}</p>` : ''}
                    </div>
                `).join('');
            } else {
                container.innerHTML = '<div class="empty-state">Aucun skill charge. Ajoutez des skills dans le dossier skills/custom/.</div>';
            }
        } catch (e) {
            container.innerHTML = '<div class="empty-state">Impossible de charger les skills.</div>';
        }
    }

    // ── Config ──────────────────────────────────────────────

    async loadConfig() {
        const container = document.getElementById('config-editor');
        if (!container) return;

        try {
            const resp = await fetch('/api/config');
            const data = await resp.json();

            // Build an interactive config editor
            container.innerHTML = this.buildConfigEditor(data);
            this.bindConfigActions(data);
        } catch (e) {
            container.innerHTML = '<div class="empty-state">Impossible de charger la configuration.</div>';
        }
    }

    buildConfigEditor(config) {
        // Extract key sections for quick-edit
        const providers = config.providers || {};
        const gateway = config.gateway || {};
        const agent = config.agent || {};

        let html = '';

        // Quick Setup section
        html += '<div class="config-section">';
        html += '<h3>Fournisseurs LLM</h3>';
        html += '<p class="config-help">Configurez au moins un fournisseur pour activer le chat.</p>';

        for (const [name, prov] of Object.entries(providers)) {
            if (!prov || typeof prov !== 'object') continue;
            const enabled = prov.enabled ? 'active' : '';
            const keyDisplay = prov.api_key || '';
            const needsKey = name !== 'ollama';

            html += `
                <div class="config-provider ${enabled}">
                    <div class="config-provider-header">
                        <label class="config-toggle">
                            <input type="checkbox" data-path="providers.${name}.enabled"
                                   ${prov.enabled ? 'checked' : ''}>
                            <span class="config-toggle-label">${this.esc(name.charAt(0).toUpperCase() + name.slice(1))}</span>
                        </label>
                        <span class="config-provider-model">${this.esc(prov.default_model || '')}</span>
                    </div>
                    ${needsKey ? `
                    <div class="config-field">
                        <label>Cle API</label>
                        <div class="config-input-group">
                            <input type="password" class="config-input"
                                   data-path="providers.${name}.api_key"
                                   value="${this.esc(keyDisplay)}"
                                   placeholder="Entrez votre cle API ${name}...">
                            <button class="btn-sm config-toggle-vis" title="Afficher/masquer">&#128065;</button>
                        </div>
                    </div>` : `
                    <div class="config-field">
                        <label>URL</label>
                        <input type="text" class="config-input"
                               data-path="providers.${name}.base_url"
                               value="${this.esc(prov.base_url || '')}"
                               placeholder="http://localhost:11434">
                    </div>`}
                </div>
            `;
        }
        html += '</div>';

        // Gateway section
        html += '<div class="config-section">';
        html += '<h3>Gateway</h3>';
        html += `
            <div class="config-field-row">
                <div class="config-field">
                    <label>Host</label>
                    <input type="text" class="config-input" data-path="gateway.host"
                           value="${this.esc(gateway.host || '0.0.0.0')}">
                </div>
                <div class="config-field">
                    <label>Port</label>
                    <input type="number" class="config-input" data-path="gateway.port"
                           value="${gateway.port || 18789}">
                </div>
            </div>
        `;
        html += '</div>';

        // Agent section
        html += '<div class="config-section">';
        html += '<h3>Agent</h3>';
        html += `
            <div class="config-field-row">
                <div class="config-field">
                    <label>Temperature</label>
                    <input type="number" class="config-input" data-path="agent.temperature"
                           value="${agent.temperature || 0.7}" step="0.1" min="0" max="2">
                </div>
                <div class="config-field">
                    <label>Max Tokens</label>
                    <input type="number" class="config-input" data-path="agent.max_tokens"
                           value="${agent.max_tokens || 4096}" step="256">
                </div>
            </div>
        `;
        html += '</div>';

        // Save button + Raw JSON toggle
        html += `
            <div class="config-actions">
                <button id="config-save" class="btn-primary">Sauvegarder</button>
                <button id="config-toggle-raw" class="btn-sm">Voir JSON brut</button>
                <span id="config-status" class="config-status"></span>
            </div>
            <div id="config-raw" class="config-raw hidden">
                <textarea id="config-raw-editor" class="config-raw-editor"
                    rows="20">${JSON.stringify(config, null, 2)}</textarea>
                <button id="config-save-raw" class="btn-sm">Sauvegarder JSON</button>
            </div>
        `;

        return html;
    }

    bindConfigActions(originalConfig) {
        // Toggle password visibility
        document.querySelectorAll('.config-toggle-vis').forEach(btn => {
            btn.addEventListener('click', () => {
                const input = btn.parentElement.querySelector('input');
                input.type = input.type === 'password' ? 'text' : 'password';
            });
        });

        // Save button
        const saveBtn = document.getElementById('config-save');
        if (saveBtn) {
            saveBtn.addEventListener('click', () => this.saveConfig());
        }

        // Toggle raw JSON
        const toggleRaw = document.getElementById('config-toggle-raw');
        if (toggleRaw) {
            toggleRaw.addEventListener('click', () => {
                const raw = document.getElementById('config-raw');
                if (raw) raw.classList.toggle('hidden');
            });
        }

        // Save raw JSON
        const saveRaw = document.getElementById('config-save-raw');
        if (saveRaw) {
            saveRaw.addEventListener('click', () => this.saveRawConfig());
        }
    }

    async saveConfig() {
        const status = document.getElementById('config-status');
        const updates = {};

        // Collect all changed values
        document.querySelectorAll('.config-input').forEach(input => {
            const path = input.dataset.path;
            if (!path) return;
            let value = input.value;
            if (input.type === 'number') value = parseFloat(value);
            updates[path] = value;
        });

        // Collect checkboxes
        document.querySelectorAll('.config-provider input[type=checkbox]').forEach(cb => {
            const path = cb.dataset.path;
            if (!path) return;
            updates[path] = cb.checked;
        });

        if (status) status.textContent = 'Sauvegarde...';

        try {
            const resp = await fetch('/api/config', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updates),
            });
            if (resp.ok) {
                if (status) {
                    status.textContent = 'Sauvegarde !';
                    status.className = 'config-status success';
                    setTimeout(() => { status.textContent = ''; }, 3000);
                }
            } else {
                if (status) {
                    status.textContent = 'Erreur de sauvegarde';
                    status.className = 'config-status error';
                }
            }
        } catch (e) {
            if (status) {
                status.textContent = 'Erreur reseau';
                status.className = 'config-status error';
            }
        }
    }

    async saveRawConfig() {
        const textarea = document.getElementById('config-raw-editor');
        if (!textarea) return;

        try {
            const raw = JSON.parse(textarea.value);
            // Flatten to dot-path keys for the PUT endpoint
            const flat = {};
            function flatten(obj, prefix) {
                for (const [k, v] of Object.entries(obj)) {
                    const key = prefix ? prefix + '.' + k : k;
                    if (v && typeof v === 'object' && !Array.isArray(v)) {
                        flatten(v, key);
                    } else {
                        flat[key] = v;
                    }
                }
            }
            flatten(raw, '');

            const resp = await fetch('/api/config', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(flat),
            });
            if (resp.ok) {
                alert('Configuration sauvegardee. Rechargez la page pour appliquer.');
            }
        } catch (e) {
            alert('JSON invalide: ' + e.message);
        }
    }

    // ── Utilities ───────────────────────────────────────────

    esc(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

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
