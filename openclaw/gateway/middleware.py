"""
Gateway Middleware: Security, Rate Limiting, Semantic Caching.
"""

import hashlib
import logging
import re
import time
from collections import defaultdict
from typing import Optional

from openclaw.config.settings import get_settings

logger = logging.getLogger("openclaw.gateway.middleware")


# ── Security Middleware ──────────────────────────────────────────────

class SecurityMiddleware:
    """
    Content-aware security layer:
    - Prompt injection detection
    - PII filtering
    - Content length validation
    - API key verification
    """

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"disregard\s+(all\s+)?prior",
        r"forget\s+everything",
        r"you\s+are\s+now\s+(?:DAN|evil|unrestricted)",
        r"system\s*:\s*override",
        r"\[INST\].*\[/INST\]",
        r"<\|im_start\|>system",
    ]

    PII_PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    }

    def __init__(self):
        self.settings = get_settings()
        self._compiled_injection = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
        self._compiled_pii = {k: re.compile(v) for k, v in self.PII_PATTERNS.items()}

    def validate_request(self, content: str, api_key: str = None) -> tuple[bool, str]:
        """Validate an incoming request. Returns (is_valid, error_message)."""
        # API key check
        if self.settings.get("gateway.security.api_key_required", False):
            valid_keys = self.settings.get("gateway.security.api_keys", [])
            if api_key not in valid_keys:
                return False, "Invalid API key"

        # Length check
        max_len = self.settings.get("gateway.security.max_prompt_length", 32000)
        if len(content) > max_len:
            return False, f"Prompt exceeds maximum length ({max_len} chars)"

        # Injection detection
        if self.settings.get("gateway.security.content_filtering", True):
            for pattern in self._compiled_injection:
                if pattern.search(content):
                    logger.warning(f"Potential prompt injection detected")
                    return False, "Request blocked: suspicious content detected"

        return True, ""

    def filter_output(self, content: str) -> str:
        """Filter sensitive data from output."""
        if not self.settings.get("gateway.security.pii_detection", False):
            return content

        filtered = content
        for pii_type, pattern in self._compiled_pii.items():
            filtered = pattern.sub(f"[{pii_type.upper()}_REDACTED]", filtered)
        return filtered


# ── Rate Limiter ─────────────────────────────────────────────────────

class RateLimiter:
    """
    Token-aware rate limiter with sliding window.
    Tracks both request count and token consumption.
    """

    def __init__(self):
        self.settings = get_settings()
        self._request_windows: dict[str, list[float]] = defaultdict(list)
        self._token_windows: dict[str, list[tuple[float, int]]] = defaultdict(list)

    def check_limit(self, client_id: str, estimated_tokens: int = 0) -> tuple[bool, dict]:
        """
        Check if a request is within rate limits.
        Returns (allowed, info_dict).
        """
        if not self.settings.get("gateway.rate_limit.enabled", True):
            return True, {}

        now = time.time()
        window = 60.0  # 1-minute window
        rpm = self.settings.get("gateway.rate_limit.requests_per_minute", 60)
        tpm = self.settings.get("gateway.rate_limit.tokens_per_minute", 100000)
        burst = self.settings.get("gateway.rate_limit.burst_multiplier", 2.0)

        # Clean old entries
        self._request_windows[client_id] = [
            t for t in self._request_windows[client_id] if now - t < window
        ]
        self._token_windows[client_id] = [
            (t, tok) for t, tok in self._token_windows[client_id] if now - t < window
        ]

        req_count = len(self._request_windows[client_id])
        token_count = sum(tok for _, tok in self._token_windows[client_id])

        info = {
            "requests_remaining": max(0, int(rpm * burst) - req_count),
            "tokens_remaining": max(0, int(tpm * burst) - token_count),
            "reset_at": now + window,
        }

        if req_count >= rpm * burst:
            return False, {**info, "reason": "Request rate limit exceeded"}

        if token_count + estimated_tokens > tpm * burst:
            return False, {**info, "reason": "Token rate limit exceeded"}

        # Record this request
        self._request_windows[client_id].append(now)
        if estimated_tokens > 0:
            self._token_windows[client_id].append((now, estimated_tokens))

        return True, info

    def record_tokens(self, client_id: str, tokens: int):
        """Record actual token usage after response."""
        self._token_windows[client_id].append((time.time(), tokens))


# ── Response Cache ───────────────────────────────────────────────────

class SemanticCache:
    """
    Response cache with exact matching and fuzzy similarity fallback.
    Uses Jaccard token similarity when exact match fails.
    Note: For true semantic similarity, integrate embeddings (sentence-transformers).
    """

    def __init__(self):
        self.settings = get_settings()
        self._cache: dict[str, dict] = {}
        self._max_entries = self.settings.get("gateway.cache.max_entries", 1000)
        self._ttl = self.settings.get("gateway.cache.ttl_seconds", 3600)
        self._similarity_threshold = self.settings.get(
            "gateway.cache.semantic_similarity_threshold", 0.92
        )

    def get(self, prompt: str, model: str = "") -> Optional[dict]:
        """Look up a cached response for a similar prompt."""
        if not self.settings.get("gateway.cache.enabled", True):
            return None

        now = time.time()
        normalized = self._normalize(prompt)
        cache_key = self._compute_key(normalized, model)

        # Exact match first
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if now - entry["timestamp"] < self._ttl:
                entry["hits"] += 1
                logger.debug(f"Cache hit (exact): {cache_key[:16]}...")
                return entry["response"]
            else:
                del self._cache[cache_key]

        # Fuzzy match fallback (token overlap)
        if self._similarity_threshold < 1.0:
            best_match = self._find_similar(normalized, model, now)
            if best_match:
                best_match["hits"] += 1
                logger.debug("Cache hit (fuzzy)")
                return best_match["response"]

        return None

    def put(self, prompt: str, model: str, response: dict):
        """Store a response in the cache."""
        if not self.settings.get("gateway.cache.enabled", True):
            return

        # Evict if full
        if len(self._cache) >= self._max_entries:
            self._evict_lru()

        normalized = self._normalize(prompt)
        cache_key = self._compute_key(normalized, model)
        self._cache[cache_key] = {
            "prompt": prompt,
            "normalized": normalized,
            "tokens": set(normalized.split()),
            "model": model,
            "response": response,
            "timestamp": time.time(),
            "hits": 0,
        }

    def invalidate(self, pattern: str = None):
        """Clear cache entries matching a pattern, or all."""
        if pattern is None:
            self._cache.clear()
        else:
            to_remove = [k for k, v in self._cache.items() if pattern in v.get("prompt", "")]
            for k in to_remove:
                del self._cache[k]

    def stats(self) -> dict:
        return {
            "entries": len(self._cache),
            "max_entries": self._max_entries,
            "total_hits": sum(e["hits"] for e in self._cache.values()),
        }

    def _normalize(self, text: str) -> str:
        """Normalize prompt for comparison."""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def _compute_key(self, normalized: str, model: str) -> str:
        return hashlib.sha256(f"{model}:{normalized}".encode()).hexdigest()

    def _find_similar(self, normalized: str, model: str, now: float) -> Optional[dict]:
        """Find a similar cached entry using Jaccard token similarity."""
        query_tokens = set(normalized.split())
        if not query_tokens:
            return None

        best_entry = None
        best_score = 0.0

        for entry in self._cache.values():
            if entry["model"] != model:
                continue
            if now - entry["timestamp"] >= self._ttl:
                continue

            entry_tokens = entry.get("tokens", set())
            if not entry_tokens:
                continue

            # Jaccard similarity: |A ∩ B| / |A ∪ B|
            intersection = len(query_tokens & entry_tokens)
            union = len(query_tokens | entry_tokens)
            score = intersection / union if union > 0 else 0.0

            if score >= self._similarity_threshold and score > best_score:
                best_score = score
                best_entry = entry

        return best_entry

    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self._cache:
            return
        oldest_key = min(self._cache, key=lambda k: self._cache[k]["timestamp"])
        del self._cache[oldest_key]
