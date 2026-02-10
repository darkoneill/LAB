"""
OpenClaw Gateway Server
High-performance API gateway with streaming, semantic caching, and multi-protocol support.
Inspired by Kong, LiteLLM, and modern AI gateway best practices.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional, AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from openclaw.config.settings import get_settings
from openclaw.gateway.middleware import RateLimiter
from openclaw.gateway.approval import ApprovalMiddleware
from openclaw.tracing import get_tracer

logger = logging.getLogger("openclaw.gateway")


# ── Request/Response Models ───────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = "user"
    content: str
    name: Optional[str] = None
    metadata: Optional[dict] = None


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    session_id: Optional[str] = None
    tools: Optional[list[str]] = None
    metadata: Optional[dict] = None


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"resp_{uuid.uuid4().hex[:12]}")
    session_id: str = ""
    content: str = ""
    role: str = "assistant"
    model: str = ""
    usage: dict = Field(default_factory=dict)
    tool_calls: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = ""
    uptime_seconds: float = 0
    active_sessions: int = 0
    memory_usage_mb: float = 0


# ── Session Manager ──────────────────────────────────────────────────

class SessionManager:
    """Manages active chat sessions with history."""

    def __init__(self):
        self.sessions: dict[str, dict] = {}

    def get_or_create(self, session_id: Optional[str] = None) -> dict:
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            session["last_active"] = time.time()
            return session

        new_id = session_id or f"session_{uuid.uuid4().hex[:12]}"
        session = {
            "id": new_id,
            "messages": [],
            "created_at": time.time(),
            "last_active": time.time(),
            "metadata": {},
        }
        self.sessions[new_id] = session
        return session

    def add_message(self, session_id: str, role: str, content: str, metadata: dict = None):
        if session_id in self.sessions:
            self.sessions[session_id]["messages"].append({
                "role": role,
                "content": content,
                "timestamp": time.time(),
                "metadata": metadata or {},
            })

    def get_history(self, session_id: str, limit: int = 50) -> list[dict]:
        if session_id in self.sessions:
            return self.sessions[session_id]["messages"][-limit:]
        return []

    def cleanup_stale(self, max_age_seconds: int = 7200):
        now = time.time()
        stale = [sid for sid, s in self.sessions.items() if now - s["last_active"] > max_age_seconds]
        for sid in stale:
            del self.sessions[sid]

    @property
    def active_count(self) -> int:
        return len(self.sessions)


# ── WebSocket Connection Manager ─────────────────────────────────────

class ConnectionManager:
    """Manages WebSocket connections for real-time streaming."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket connected: {client_id}")

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)
        logger.info(f"WebSocket disconnected: {client_id}")

    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)

    async def broadcast(self, message: dict):
        for ws in self.active_connections.values():
            try:
                await ws.send_json(message)
            except Exception:
                pass

    @property
    def count(self) -> int:
        return len(self.active_connections)


# ── Gateway Server ───────────────────────────────────────────────────

class GatewayServer:
    """
    Main Gateway Server orchestrating all components.
    Features:
    - REST + WebSocket endpoints
    - SSE streaming
    - Semantic caching
    - Token-aware rate limiting
    - Model failover
    - Session management
    - Health monitoring
    """

    def __init__(self, agent_brain=None, memory_manager=None, skill_router=None):
        self.settings = get_settings()
        self.app = FastAPI(
            title="OpenClaw Gateway",
            version="1.0.0",
            description="AI Assistant Gateway with streaming, caching, and multi-model support",
        )
        self.agent = agent_brain
        self.memory = memory_manager
        self.skills = skill_router
        self.sessions = SessionManager()
        self.ws_manager = ConnectionManager()
        self.rate_limiter = RateLimiter()
        self.tracer = get_tracer()
        self.approval = ApprovalMiddleware(ws_manager=self.ws_manager)
        self.start_time = time.time()
        self._setup_middleware()
        self._setup_routes()

    def _setup_middleware(self):
        """Configure CORS, security, and performance middleware."""
        origins = self.settings.get("gateway.cors_origins", ["*"])
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Register all API routes."""

        # ── Health & Info ─────────────────────────────────
        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            mem = 0
            try:
                import psutil
                mem = psutil.Process().memory_info().rss / 1024 / 1024
            except (ImportError, Exception):
                pass
            return HealthResponse(
                status="healthy",
                version=self.settings.get("app.version", "1.0.0"),
                uptime_seconds=round(time.time() - self.start_time, 2),
                active_sessions=self.sessions.active_count,
                memory_usage_mb=round(mem, 2),
            )

        @self.app.get("/api/info")
        async def info():
            return {
                "name": self.settings.get("app.name"),
                "version": self.settings.get("app.version"),
                "codename": self.settings.get("app.codename"),
                "providers": self._get_active_providers(),
                "features": {
                    "streaming": self.settings.get("gateway.streaming.enabled", True),
                    "cache": self.settings.get("gateway.cache.enabled", True),
                    "memory": self.settings.get("memory.enabled", True),
                    "skills": self.settings.get("skills.enabled", True),
                },
            }

        # ── Chat API ─────────────────────────────────────
        @self.app.post("/api/chat")
        async def chat(request: ChatRequest, req: Request):
            # Get client identifier for rate limiting
            client_id = req.headers.get("X-Client-Id", req.client.host if req.client else "unknown")

            # Estimate tokens (rough: 4 chars = 1 token)
            total_content = "".join(m.content for m in request.messages)
            estimated_tokens = len(total_content) // 4 + (request.max_tokens or 1000)

            # Check rate limit
            allowed, rate_info = self.rate_limiter.check_limit(client_id, estimated_tokens)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"error": rate_info.get("reason", "Rate limit exceeded")},
                    headers=self._rate_limit_headers(rate_info),
                )

            session = self.sessions.get_or_create(request.session_id)

            # Store user message
            for msg in request.messages:
                self.sessions.add_message(session["id"], msg.role, msg.content, msg.metadata)

            if request.stream:
                return StreamingResponse(
                    self._stream_response(session, request),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Session-Id": session["id"],
                        **self._rate_limit_headers(rate_info),
                    },
                )

            # Non-streaming response
            response = await self._generate_response(session, request)
            return JSONResponse(
                content=response.model_dump(),
                headers=self._rate_limit_headers(rate_info),
            )

        @self.app.post("/api/chat/simple")
        async def chat_simple(request: Request):
            """Simplified chat endpoint - just send text, get text back."""
            # Get client identifier for rate limiting
            client_id = request.headers.get("X-Client-Id", request.client.host if request.client else "unknown")

            body = await request.json()
            message = body.get("message", body.get("content", ""))
            session_id = body.get("session_id", None)

            if not message:
                raise HTTPException(400, "Message is required")

            # Estimate tokens and check rate limit
            estimated_tokens = len(message) // 4 + 1000
            allowed, rate_info = self.rate_limiter.check_limit(client_id, estimated_tokens)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"error": rate_info.get("reason", "Rate limit exceeded")},
                    headers=self._rate_limit_headers(rate_info),
                )

            session = self.sessions.get_or_create(session_id)
            self.sessions.add_message(session["id"], "user", message)

            chat_req = ChatRequest(
                messages=[ChatMessage(role="user", content=message)],
                session_id=session["id"],
            )
            response = await self._generate_response(session, chat_req)
            return JSONResponse(
                content={"reply": response.content, "session_id": session["id"]},
                headers=self._rate_limit_headers(rate_info),
            )

        # ── Sessions ─────────────────────────────────────
        @self.app.get("/api/sessions")
        async def list_sessions():
            return {
                "sessions": [
                    {
                        "id": s["id"],
                        "created_at": s["created_at"],
                        "last_active": s["last_active"],
                        "message_count": len(s["messages"]),
                    }
                    for s in self.sessions.sessions.values()
                ]
            }

        @self.app.get("/api/sessions/{session_id}/history")
        async def get_session_history(session_id: str, limit: int = 50):
            history = self.sessions.get_history(session_id, limit)
            if not history:
                raise HTTPException(404, "Session not found")
            return {"session_id": session_id, "messages": history}

        @self.app.delete("/api/sessions/{session_id}")
        async def delete_session(session_id: str):
            if session_id in self.sessions.sessions:
                del self.sessions.sessions[session_id]
                return {"deleted": True}
            raise HTTPException(404, "Session not found")

        # ── Memory API ───────────────────────────────────
        @self.app.get("/api/memory/search")
        async def memory_search(query: str, top_k: int = 10):
            if not self.memory:
                raise HTTPException(503, "Memory system not initialized")
            results = await self.memory.search(query, top_k=top_k)
            return {"query": query, "results": results}

        @self.app.get("/api/memory/categories")
        async def memory_categories():
            if not self.memory:
                raise HTTPException(503, "Memory system not initialized")
            categories = await self.memory.list_categories()
            return {"categories": categories}

        # ── Skills API ───────────────────────────────────
        @self.app.get("/api/skills")
        async def list_skills():
            if not self.skills:
                return {"skills": []}
            return {"skills": self.skills.list_skills()}

        # ── Models API ───────────────────────────────────
        @self.app.get("/api/models")
        async def list_models():
            return {"models": self._get_available_models()}

        # ── Config API ───────────────────────────────────
        @self.app.get("/api/config")
        async def get_config():
            """Return safe (no secrets) config."""
            cfg = self.settings.all()
            # Redact API keys
            for provider in cfg.get("providers", {}).values():
                if isinstance(provider, dict) and "api_key" in provider:
                    key = provider["api_key"]
                    provider["api_key"] = f"***{key[-4:]}" if key and len(key) > 4 else ""
            return cfg

        @self.app.put("/api/config")
        async def update_config(request: Request):
            body = await request.json()
            for dotpath, value in body.items():
                self.settings.set(dotpath, value, persist=True)
            return {"updated": list(body.keys())}

        # ── Tracing / Observability API ─────────────────
        @self.app.get("/api/traces")
        async def list_traces(
            session_id: str = None,
            status: str = None,
            limit: int = 50,
            offset: int = 0,
        ):
            return {
                "traces": self.tracer.list_traces(session_id, status, limit, offset),
                "stats": self.tracer.get_stats(),
            }

        # Fixed: static routes BEFORE parameterized routes to avoid conflict
        @self.app.get("/api/traces/stats")
        async def trace_stats():
            return self.tracer.get_stats()

        @self.app.get("/api/traces/search/{query}")
        async def search_traces(query: str, limit: int = 20):
            return {"results": self.tracer.search_traces(query, limit)}

        @self.app.get("/api/traces/{trace_id}")
        async def get_trace(trace_id: str):
            trace = self.tracer.get_trace(trace_id)
            if not trace:
                raise HTTPException(404, "Trace not found")
            return trace

        # ── Swarm API ───────────────────────────────────
        @self.app.get("/api/swarm/profiles")
        async def swarm_profiles():
            from openclaw.agent.swarm import AGENT_PROFILES
            return {
                "profiles": {
                    role: {
                        "name": p["name"],
                        "sandbox_access": p["sandbox_access"],
                        "tools": p["tools"],
                    }
                    for role, p in AGENT_PROFILES.items()
                }
            }

        @self.app.post("/api/swarm/execute")
        async def swarm_execute(request: Request):
            body = await request.json()
            task = body.get("task", "")
            roles = body.get("roles", None)
            session_id = body.get("session_id", "")
            if not task:
                raise HTTPException(400, "Task is required")
            if not self.agent:
                raise HTTPException(503, "Agent brain not initialized")
            from openclaw.agent.swarm import SwarmOrchestrator
            swarm = SwarmOrchestrator(self.agent)
            result = await swarm.execute_swarm(task, roles, session_id)
            return {
                "success": result.success,
                "code": result.code,
                "review": result.review,
                "critic_verdict": result.critic_verdict,
                "validated": result.validated,
                "final_output": result.final_output,
                "iterations": result.iterations,
                "agents_used": result.agents_used,
            }

        # ── Approval API (Human-in-the-Loop) ────────────
        @self.app.get("/api/approvals/pending")
        async def pending_approvals():
            return {"pending": self.approval.get_pending()}

        @self.app.post("/api/approvals/{approval_id}")
        async def resolve_approval(approval_id: str, request: Request):
            body = await request.json()
            approved = body.get("approved", False)
            decided_by = body.get("decided_by", "user")
            success = self.approval.resolve_approval(approval_id, approved, decided_by)
            if not success:
                raise HTTPException(404, "Approval request not found or already resolved")
            return {"resolved": True, "approved": approved}

        @self.app.get("/api/approvals/history")
        async def approval_history(limit: int = 50):
            return {"history": self.approval.get_history(limit)}

        # ── Whisper Mode: Batch Approval + Trust ─────────
        @self.app.post("/api/approvals/batch")
        async def batch_approval(request: Request):
            """Resolve multiple approvals at once (Whisper Mode)."""
            body = await request.json()
            approval_ids = body.get("approval_ids", [])
            approved = body.get("approved", False)
            decided_by = body.get("decided_by", "user")
            trust_minutes = body.get("trust_minutes", 0)
            if not approval_ids:
                raise HTTPException(400, "approval_ids list is required")
            result = self.approval.resolve_batch(
                approval_ids, approved, decided_by, trust_minutes
            )
            return result

        @self.app.get("/api/approvals/trusted")
        async def list_trusted():
            """List tools with active temporary trust."""
            return {"trusted": self.approval.get_trusted()}

        @self.app.post("/api/approvals/trust")
        async def grant_trust(request: Request):
            """Grant temporary trust to a tool."""
            body = await request.json()
            tool_name = body.get("tool_name", "")
            server_name = body.get("server_name", "")
            duration = body.get("duration_minutes", 0)
            if not tool_name:
                raise HTTPException(400, "tool_name is required")
            expiry = self.approval.grant_trust(tool_name, server_name, duration)
            return {"trusted": True, "expires_at": expiry}

        @self.app.delete("/api/approvals/trust")
        async def revoke_trust(tool_name: str = "", server_name: str = ""):
            """Revoke temporary trust (use query params, not body)."""
            count = self.approval.revoke_trust(tool_name, server_name)
            return {"revoked": count}

        # ── WebSocket ────────────────────────────────────
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            await self.ws_manager.connect(websocket, client_id)
            session = self.sessions.get_or_create(client_id)

            try:
                while True:
                    data = await websocket.receive_json()
                    msg_type = data.get("type", "message")

                    if msg_type == "approval_response":
                        # Handle approval decision from UI
                        approval_id = data.get("approval_id", "")
                        approved = data.get("approved", False)
                        trust_minutes = data.get("trust_minutes", 0)
                        # Capture tool info BEFORE resolving (avoids race with _history)
                        pending_req = self.approval._pending.get(approval_id)
                        self.approval.resolve_approval(
                            approval_id, approved, decided_by=client_id
                        )
                        # Grant trust if requested and approved
                        if approved and trust_minutes > 0 and pending_req:
                            self.approval.grant_trust(
                                pending_req.tool_name,
                                pending_req.server_name,
                                trust_minutes,
                            )
                        await self.ws_manager.send_message(client_id, {
                            "type": "approval_resolved",
                            "approval_id": approval_id,
                            "approved": approved,
                        })

                    elif msg_type == "batch_approval":
                        # Whisper Mode: batch approve/deny multiple operations
                        approval_ids = data.get("approval_ids", [])
                        approved = data.get("approved", False)
                        trust_minutes = data.get("trust_minutes", 0)
                        result = self.approval.resolve_batch(
                            approval_ids, approved, client_id, trust_minutes
                        )
                        await self.ws_manager.send_message(client_id, {
                            "type": "batch_resolved",
                            **result,
                        })

                    elif msg_type == "message":
                        content = data.get("content", "")
                        self.sessions.add_message(session["id"], "user", content)

                        # Stream response via WebSocket
                        await self.ws_manager.send_message(client_id, {
                            "type": "start",
                            "session_id": session["id"],
                        })

                        chat_req = ChatRequest(
                            messages=[ChatMessage(role="user", content=content)],
                            session_id=session["id"],
                            stream=True,
                        )

                        full_response = ""
                        async for chunk in self._generate_stream(session, chat_req):
                            full_response += chunk
                            await self.ws_manager.send_message(client_id, {
                                "type": "chunk",
                                "content": chunk,
                            })

                        self.sessions.add_message(session["id"], "assistant", full_response)
                        await self.ws_manager.send_message(client_id, {
                            "type": "end",
                            "content": full_response,
                        })

                    elif msg_type == "ping":
                        await self.ws_manager.send_message(client_id, {"type": "pong"})

            except WebSocketDisconnect:
                self.ws_manager.disconnect(client_id)

    async def _generate_response(self, session: dict, request: ChatRequest) -> ChatResponse:
        """Generate a full response using the agent brain."""
        if not self.agent:
            return ChatResponse(
                session_id=session["id"],
                content="Agent brain not initialized. Please configure a provider.",
                model="none",
            )

        # Build context from session history + memory
        context_messages = self.sessions.get_history(session["id"])
        memory_context = ""
        if self.memory:
            last_msg = request.messages[-1].content if request.messages else ""
            memory_results = await self.memory.search(last_msg, top_k=5)
            if memory_results:
                memory_context = "\n".join([r.get("content", "") for r in memory_results])

        response = await self.agent.generate(
            messages=context_messages,
            memory_context=memory_context,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # Store assistant response
        self.sessions.add_message(session["id"], "assistant", response.get("content", ""))

        # Store in memory
        if self.memory and request.messages:
            await self.memory.store_interaction(
                user_message=request.messages[-1].content,
                assistant_response=response.get("content", ""),
                session_id=session["id"],
            )

        return ChatResponse(
            session_id=session["id"],
            content=response.get("content", ""),
            model=response.get("model", ""),
            usage=response.get("usage", {}),
            tool_calls=response.get("tool_calls", []),
        )

    async def _stream_response(self, session: dict, request: ChatRequest) -> AsyncGenerator[str, None]:
        """SSE streaming response."""
        async for chunk in self._generate_stream(session, request):
            yield f"data: {json.dumps({'content': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    async def _generate_stream(self, session: dict, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Generate streaming chunks from the agent."""
        if not self.agent:
            yield "Agent not initialized."
            return

        context_messages = self.sessions.get_history(session["id"])
        memory_context = ""
        if self.memory:
            last_msg = request.messages[-1].content if request.messages else ""
            memory_results = await self.memory.search(last_msg, top_k=5)
            if memory_results:
                memory_context = "\n".join([r.get("content", "") for r in memory_results])

        full_response = ""
        async for chunk in self.agent.generate_stream(
            messages=context_messages,
            memory_context=memory_context,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ):
            full_response += chunk
            yield chunk

        self.sessions.add_message(session["id"], "assistant", full_response)
        if self.memory and request.messages:
            await self.memory.store_interaction(
                user_message=request.messages[-1].content,
                assistant_response=full_response,
                session_id=session["id"],
            )

    def _get_active_providers(self) -> list[str]:
        providers = []
        for name in ["anthropic", "openai", "ollama", "custom"]:
            if self.settings.get(f"providers.{name}.enabled", False):
                providers.append(name)
        return providers

    def _get_available_models(self) -> list[dict]:
        models = []
        for provider_name in ["anthropic", "openai", "ollama", "custom"]:
            if self.settings.get(f"providers.{provider_name}.enabled", False):
                provider_models = self.settings.get(f"providers.{provider_name}.models", [])
                for m in provider_models:
                    models.append({
                        "provider": provider_name,
                        "id": m.get("id", ""),
                        "name": m.get("name", ""),
                        "max_tokens": m.get("max_tokens", 4096),
                        "context_window": m.get("context_window", 128000),
                    })
        return models

    def _rate_limit_headers(self, rate_info: dict) -> dict:
        """Generate X-RateLimit-* headers from rate info."""
        headers = {}
        if rate_info:
            if "requests_remaining" in rate_info:
                headers["X-RateLimit-Remaining"] = str(rate_info["requests_remaining"])
            if "tokens_remaining" in rate_info:
                headers["X-RateLimit-Tokens-Remaining"] = str(rate_info["tokens_remaining"])
            if "reset_at" in rate_info:
                headers["X-RateLimit-Reset"] = str(int(rate_info["reset_at"]))
        return headers

    async def _session_cleanup_loop(self):
        """Background task to clean up stale sessions periodically."""
        while True:
            await asyncio.sleep(300)  # Run every 5 minutes
            try:
                before = self.sessions.active_count
                self.sessions.cleanup_stale(max_age_seconds=7200)  # 2 hours
                after = self.sessions.active_count
                if before > after:
                    logger.info(f"Cleaned up {before - after} stale sessions")
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")

    async def start(self, host: str = None, port: int = None):
        """Start the gateway server."""
        import uvicorn
        host = host or self.settings.get("gateway.host", "127.0.0.1")
        port = port or self.settings.get("gateway.port", 18789)
        logger.info(f"Starting OpenClaw Gateway on {host}:{port}")

        # Start background cleanup task
        cleanup_task = asyncio.create_task(self._session_cleanup_loop())

        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info",
            ws_ping_interval=self.settings.get("gateway.streaming.heartbeat_interval", 15),
        )
        server = uvicorn.Server(config)
        try:
            await server.serve()
        finally:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
