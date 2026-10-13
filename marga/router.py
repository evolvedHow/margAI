"""Minimal routing engine with usage tracking."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol, Callable, Any


@dataclass
class Backend:
    """LLM backend configuration."""
    name: str
    base_url: str
    api_key: str = ""
    models: list[str] = None
    
    def __post_init__(self):
        if self.models is None:
            self.models = []


@dataclass
class UsageStats:
    """Track backend usage for routing decisions."""
    call_count: int = 0
    total_tokens: int = 0
    last_used: float = 0.0
    
    def record(self, tokens: int = 0):
        """Record a call to this backend."""
        self.call_count += 1
        self.total_tokens += tokens
        self.last_used = time.time()


class RouterFn(Protocol):
    """Router function signature."""
    def __call__(self, ctx: Any) -> str: ...


class Router:
    """Smart routing with usage tracking.
    
    Routes requests to backends based on:
    - Least used (default)
    - Cost optimization
    - Custom logic via registered router functions
    """
    
    def __init__(self, backends: dict[str, Backend]):
        self.backends = backends
        self.stats: dict[str, UsageStats] = defaultdict(UsageStats)
        self._router_fn: RouterFn | None = None
    
    def set_router(self, fn: RouterFn):
        """Register custom routing function."""
        self._router_fn = fn
    
    def route(self, ctx: Any) -> tuple[Backend, str]:
        """Determine which backend and model to use.
        
        Returns:
            (backend, model) tuple
        """
        # Use custom router if registered
        if self._router_fn:
            target = self._router_fn(ctx)
            return self._parse_target(target)
        
        # Default: least used backend
        return self._route_least_used(ctx)
    
    def _parse_target(self, target: str) -> tuple[Backend, str]:
        """Parse 'backend/model' string into (Backend, model)."""
        if "/" in target:
            backend_name, model = target.split("/", 1)
        else:
            # If no slash, treat as model and pick least-used backend
            backend_name = self._least_used_backend()
            model = target
        
        if backend_name not in self.backends:
            raise ValueError(f"Unknown backend: {backend_name}")
        
        return self.backends[backend_name], model
    
    def _route_least_used(self, ctx: Any) -> tuple[Backend, str]:
        """Route to the least-used backend."""
        backend_name = self._least_used_backend()
        backend = self.backends[backend_name]
        
        # Pick first available model
        model = backend.models[0] if backend.models else "default"
        
        return backend, model
    
    def _least_used_backend(self) -> str:
        """Find backend with fewest calls."""
        min_calls = float('inf')
        chosen = None
        
        for name in self.backends:
            calls = self.stats[name].call_count
            if calls < min_calls:
                min_calls = calls
                chosen = name
        
        return chosen or next(iter(self.backends))
    
    def record_usage(self, backend_name: str, tokens: int = 0):
        """Record usage stats for a backend."""
        self.stats[backend_name].record(tokens)
    
    def get_stats(self) -> dict[str, dict]:
        """Get current usage statistics."""
        return {
            name: {
                "calls": stats.call_count,
                "tokens": stats.total_tokens,
                "last_used": stats.last_used
            }
            for name, stats in self.stats.items()
        }
