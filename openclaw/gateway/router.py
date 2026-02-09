"""
Intelligent Request Router with model failover and load balancing.
"""

import asyncio
import logging
import random
import time
from typing import Optional
from dataclasses import dataclass, field

from openclaw.config.settings import get_settings

logger = logging.getLogger("openclaw.gateway.router")


@dataclass
class ProviderHealth:
    name: str
    healthy: bool = True
    last_error: float = 0
    error_count: int = 0
    avg_latency_ms: float = 0
    total_requests: int = 0
    total_tokens: int = 0
    _latencies: list = field(default_factory=list)

    def record_success(self, latency_ms: float, tokens: int = 0):
        self.healthy = True
        self.error_count = 0
        self.total_requests += 1
        self.total_tokens += tokens
        self._latencies.append(latency_ms)
        if len(self._latencies) > 100:
            self._latencies = self._latencies[-100:]
        self.avg_latency_ms = sum(self._latencies) / len(self._latencies)

    def record_failure(self):
        self.error_count += 1
        self.last_error = time.time()
        self.total_requests += 1
        if self.error_count >= 3:
            self.healthy = False

    def should_retry(self) -> bool:
        """Check if enough time has passed to retry an unhealthy provider."""
        if self.healthy:
            return True
        # Cap exponent to prevent overflow (max backoff = 64 seconds)
        capped_errors = min(self.error_count, 6)
        backoff = min(60, 2 ** capped_errors)
        return (time.time() - self.last_error) > backoff


class RequestRouter:
    """
    Routes requests to LLM providers with:
    - Automatic failover
    - Health-based routing
    - Latency-aware load balancing
    - Token budget tracking
    """

    def __init__(self):
        self.settings = get_settings()
        self.provider_health: dict[str, ProviderHealth] = {}
        self._init_providers()

    def _init_providers(self):
        for name in ["anthropic", "openai", "ollama", "custom"]:
            if self.settings.get(f"providers.{name}.enabled", False):
                self.provider_health[name] = ProviderHealth(name=name)

    def resolve_model(self, requested_model: Optional[str] = None) -> tuple[str, str]:
        """
        Resolve a model request to (provider_name, model_id).
        Supports formats: 'provider/model', 'model_id', or None (default).
        """
        if requested_model and "/" in requested_model:
            provider, model = requested_model.split("/", 1)
            if provider in self.provider_health:
                return provider, model

        # Search across providers for the model
        if requested_model:
            for provider_name, health in self.provider_health.items():
                if not health.healthy and not health.should_retry():
                    continue
                models = self.settings.get(f"providers.{provider_name}.models", [])
                for m in models:
                    if m.get("id") == requested_model or m.get("name", "").lower() == requested_model.lower():
                        return provider_name, m["id"]

        # Fall back to default
        return self._get_default_provider()

    def _get_default_provider(self) -> tuple[str, str]:
        """Get the best available default provider and model."""
        # Priority: anthropic > openai > ollama > custom
        for name in ["anthropic", "openai", "ollama", "custom"]:
            health = self.provider_health.get(name)
            if health and (health.healthy or health.should_retry()):
                default_model = self.settings.get(f"providers.{name}.default_model", "")
                if default_model:
                    return name, default_model

        raise RuntimeError("No healthy LLM provider available")

    def get_failover(self, failed_provider: str, failed_model: str) -> Optional[tuple[str, str]]:
        """Get an alternative provider after a failure."""
        for name, health in self.provider_health.items():
            if name == failed_provider:
                continue
            if health.healthy or health.should_retry():
                default_model = self.settings.get(f"providers.{name}.default_model", "")
                if default_model:
                    logger.warning(f"Failing over from {failed_provider} to {name}")
                    return name, default_model
        return None

    def record_success(self, provider: str, latency_ms: float, tokens: int = 0):
        if provider in self.provider_health:
            self.provider_health[provider].record_success(latency_ms, tokens)

    def record_failure(self, provider: str):
        if provider in self.provider_health:
            self.provider_health[provider].record_failure()

    def get_stats(self) -> dict:
        return {
            name: {
                "healthy": h.healthy,
                "avg_latency_ms": round(h.avg_latency_ms, 2),
                "total_requests": h.total_requests,
                "total_tokens": h.total_tokens,
                "error_count": h.error_count,
            }
            for name, h in self.provider_health.items()
        }
