/**
 * Mission Control - Trace visualization & approval handling.
 * Renders the agent's decision tree in real-time.
 */
(function () {
  'use strict';

  const API = window.location.origin;
  let currentApprovalId = null;

  // ── Span kind icons & colors ──────────────────────────────────
  const SPAN_ICONS = {
    request:    { icon: '\u25B6', color: '#61afef' },  // blue
    retrieval:  { icon: '\u2605', color: '#c678dd' },  // purple
    llm_call:   { icon: '\u2726', color: '#e5c07b' },  // yellow
    tool_exec:  { icon: '\u2699', color: '#98c379' },  // green
    self_heal:  { icon: '\u267B', color: '#e06c75' },  // red
    delegation: { icon: '\u21C4', color: '#56b6c2' },  // teal
    mcp_call:   { icon: '\u2693', color: '#d19a66' },  // orange
    approval:   { icon: '\u26A0', color: '#e06c75' },  // red
    response:   { icon: '\u2714', color: '#98c379' },  // green
  };

  const STATUS_CLASS = {
    ok:      'span-ok',
    error:   'span-error',
    timeout: 'span-timeout',
  };

  // ── Load traces list ──────────────────────────────────────────
  async function loadTraces() {
    try {
      const res = await fetch(API + '/api/traces?limit=50');
      const data = await res.json();
      renderTracesList(data.traces || []);
      renderStats(data.stats || {});
    } catch (e) {
      console.error('Failed to load traces:', e);
    }
  }

  function renderStats(stats) {
    const el = (id) => document.getElementById(id);
    if (el('stat-total'))  el('stat-total').textContent = stats.total_traces || 0;
    if (el('stat-active')) el('stat-active').textContent = stats.active_traces || 0;
    if (el('stat-avg'))    el('stat-avg').textContent = (stats.avg_duration_ms || 0) + 'ms';
    if (el('stat-errors')) el('stat-errors').textContent = stats.errors || 0;
  }

  function renderTracesList(traces) {
    const container = document.getElementById('traces-list');
    if (!container) return;

    if (traces.length === 0) {
      container.innerHTML = '<div class="traces-empty">Aucune trace enregistree.</div>';
      return;
    }

    container.innerHTML = traces.map(t => {
      const statusClass = t.status === 'completed' ? 'trace-ok' :
                          t.status === 'error' ? 'trace-error' : 'trace-active';
      const time = new Date(t.start_time * 1000).toLocaleTimeString('fr-FR');
      const duration = t.duration_ms ? `${t.duration_ms}ms` : 'en cours...';
      const input = (t.user_input || '').substring(0, 60);

      return `
        <div class="trace-item ${statusClass}" data-trace-id="${t.trace_id}">
          <div class="trace-item-header">
            <span class="trace-time">${time}</span>
            <span class="trace-duration">${duration}</span>
            <span class="trace-spans">${t.span_count || 0} spans</span>
          </div>
          <div class="trace-item-input">${escapeHtml(input)}</div>
        </div>`;
    }).join('');

    // Click handlers
    container.querySelectorAll('.trace-item').forEach(el => {
      el.addEventListener('click', () => loadTraceDetail(el.dataset.traceId));
    });
  }

  // ── Load trace detail (span tree) ─────────────────────────────
  async function loadTraceDetail(traceId) {
    const container = document.getElementById('trace-detail');
    if (!container) return;

    container.innerHTML = '<div class="trace-loading">Chargement...</div>';

    try {
      const res = await fetch(API + '/api/traces/' + traceId);
      const trace = await res.json();
      renderSpanTree(container, trace);
    } catch (e) {
      container.innerHTML = '<div class="trace-error">Erreur de chargement.</div>';
    }
  }

  function renderSpanTree(container, trace) {
    const header = `
      <div class="trace-detail-header">
        <h3>Trace: ${trace.trace_id}</h3>
        <span class="trace-detail-status trace-${trace.status}">${trace.status}</span>
        ${trace.duration_ms ? `<span class="trace-detail-duration">${trace.duration_ms}ms</span>` : ''}
      </div>
      <div class="trace-detail-input">
        <strong>Input:</strong> ${escapeHtml((trace.user_input || '').substring(0, 200))}
      </div>`;

    const spans = (trace.spans || []).map(span => {
      const meta = SPAN_ICONS[span.kind] || { icon: '\u25CB', color: '#abb2bf' };
      const sClass = STATUS_CLASS[span.status] || '';
      const duration = span.duration_ms ? `${span.duration_ms}ms` : '...';
      const indent = span.parent_span_id ? 'span-child' : '';
      const attrs = Object.entries(span.attributes || {})
        .slice(0, 5)
        .map(([k, v]) => `<span class="span-attr">${escapeHtml(k)}: ${escapeHtml(String(v).substring(0, 80))}</span>`)
        .join('');

      return `
        <div class="span-node ${sClass} ${indent}">
          <div class="span-header">
            <span class="span-icon" style="color:${meta.color}">${meta.icon}</span>
            <span class="span-kind">${span.kind}</span>
            <span class="span-name">${escapeHtml(span.name || '')}</span>
            <span class="span-duration">${duration}</span>
            <span class="span-status-dot ${sClass}"></span>
          </div>
          ${attrs ? `<div class="span-attrs">${attrs}</div>` : ''}
          ${span.events && span.events.length ? renderEvents(span.events) : ''}
        </div>`;
    }).join('');

    const response = trace.final_response
      ? `<div class="trace-detail-response">
           <strong>Response:</strong> ${escapeHtml(trace.final_response.substring(0, 300))}
         </div>`
      : '';

    container.innerHTML = header + '<div class="span-tree">' + spans + '</div>' + response;
  }

  function renderEvents(events) {
    return '<div class="span-events">' +
      events.map(e =>
        `<div class="span-event"><span class="event-name">${escapeHtml(e.name)}</span></div>`
      ).join('') + '</div>';
  }

  // ── Approval handling ─────────────────────────────────────────
  // Track pending approvals for batch mode (Whisper Mode)
  let pendingApprovals = [];

  function showApprovalBanner(data) {
    const banner = document.getElementById('approval-banner');
    const text = document.getElementById('approval-text');
    if (!banner || !text) return;

    currentApprovalId = data.id;

    // Add to pending list for batch mode
    if (!pendingApprovals.find(a => a.id === data.id)) {
      pendingApprovals.push({
        id: data.id,
        tool_name: data.tool_name,
        description: data.description,
        safety_level: data.safety_level,
      });
    }

    // Update banner text
    if (pendingApprovals.length > 1) {
      text.textContent = pendingApprovals.length + ' operations en attente - Mode Whisper disponible';
    } else {
      text.textContent = data.description || "L'agent veut executer: " + data.tool_name;
    }
    banner.classList.remove('hidden');

    // Show batch button if multiple pending
    updateBatchButton();
  }

  function hideApprovalBanner() {
    const banner = document.getElementById('approval-banner');
    if (banner) banner.classList.add('hidden');
    currentApprovalId = null;
    pendingApprovals = [];
    updateBatchButton();
  }

  function updateBatchButton() {
    var batchBtn = document.getElementById('approval-batch');
    if (!batchBtn) return;
    if (pendingApprovals.length > 1) {
      batchBtn.classList.remove('hidden');
      batchBtn.textContent = 'Tout autoriser (' + pendingApprovals.length + ')';
    } else {
      batchBtn.classList.add('hidden');
    }
  }

  async function resolveApproval(approved) {
    if (!currentApprovalId) return;
    var trustMinutes = getTrustDuration();
    try {
      if (window.ws && window.ws.readyState === WebSocket.OPEN) {
        window.ws.send(JSON.stringify({
          type: 'approval_response',
          approval_id: currentApprovalId,
          approved: approved,
          trust_minutes: trustMinutes,
        }));
      } else {
        await fetch(API + '/api/approvals/' + currentApprovalId, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ approved: approved, trust_minutes: trustMinutes }),
        });
      }
    } catch (e) {
      console.error('Failed to resolve approval:', e);
    }
    hideApprovalBanner();
  }

  async function resolveBatchApproval() {
    if (pendingApprovals.length === 0) return;
    var ids = pendingApprovals.map(function(a) { return a.id; });
    var trustMinutes = getTrustDuration();
    try {
      if (window.ws && window.ws.readyState === WebSocket.OPEN) {
        window.ws.send(JSON.stringify({
          type: 'batch_approval',
          approval_ids: ids,
          approved: true,
          trust_minutes: trustMinutes,
        }));
      } else {
        await fetch(API + '/api/approvals/batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            approval_ids: ids,
            approved: true,
            trust_minutes: trustMinutes,
          }),
        });
      }
    } catch (e) {
      console.error('Failed to batch approve:', e);
    }
    hideApprovalBanner();
  }

  function getTrustDuration() {
    var select = document.getElementById('trust-duration');
    return select ? parseInt(select.value, 10) || 0 : 0;
  }

  // ── WebSocket listener for live updates ───────────────────────
  function setupWsListeners() {
    const origOnMessage = window.wsOnMessage;
    window.wsOnMessage = function (event) {
      if (origOnMessage) origOnMessage(event);
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'approval_request') {
          showApprovalBanner(data);
        } else if (data.type === 'approval_resolved') {
          hideApprovalBanner();
        }
      } catch (e) { /* ignore non-JSON */ }
    };
  }

  // ── Helpers ───────────────────────────────────────────────────
  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // ── Init ──────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', function () {
    // Refresh button
    const refreshBtn = document.getElementById('traces-refresh');
    if (refreshBtn) refreshBtn.addEventListener('click', loadTraces);

    // Approval buttons
    const acceptBtn = document.getElementById('approval-accept');
    const denyBtn = document.getElementById('approval-deny');
    const batchBtn = document.getElementById('approval-batch');
    if (acceptBtn) acceptBtn.addEventListener('click', () => resolveApproval(true));
    if (denyBtn) denyBtn.addEventListener('click', () => resolveApproval(false));
    if (batchBtn) batchBtn.addEventListener('click', () => resolveBatchApproval());

    // Auto-load traces when switching to view
    const observer = new MutationObserver(function () {
      const tracesView = document.getElementById('view-traces');
      if (tracesView && tracesView.classList.contains('active')) {
        loadTraces();
      }
    });
    const content = document.getElementById('content');
    if (content) observer.observe(content, { subtree: true, attributes: true, attributeFilter: ['class'] });

    setupWsListeners();
  });
})();
