"""Before/after hooks for request/response processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class HookContext:
    """Context passed to hooks.
    
    Attributes:
        messages: The conversation messages
        model: Requested model
        stream: Whether streaming is enabled
        metadata: Arbitrary metadata for passing data between hooks
    """
    messages: list[dict[str, Any]]
    model: str
    stream: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def estimated_tokens(self) -> int:
        """Rough token estimate (4 chars = 1 token)."""
        total_chars = sum(
            len(msg.get("content", ""))
            for msg in self.messages
        )
        return total_chars // 4


BeforeHook = Callable[[HookContext], None]
AfterHook = Callable[[HookContext, dict[str, Any]], dict[str, Any]]


class HookManager:
    """Manages before/after hooks."""
    
    def __init__(self):
        self._before_hooks: list[BeforeHook] = []
        self._after_hooks: list[AfterHook] = []
    
    def add_before(self, fn: BeforeHook):
        """Register a before hook.
        
        Before hooks can modify the HookContext (messages, model, etc)
        before the request is sent to the LLM.
        """
        self._before_hooks.append(fn)
    
    def add_after(self, fn: AfterHook):
        """Register an after hook.
        
        After hooks receive the HookContext and response, and can
        modify the response before returning to the client.
        """
        self._after_hooks.append(fn)
    
    def run_before(self, ctx: HookContext):
        """Execute all before hooks in order."""
        for hook in self._before_hooks:
            hook(ctx)
    
    def run_after(self, ctx: HookContext, response: dict[str, Any]) -> dict[str, Any]:
        """Execute all after hooks in order."""
        for hook in self._after_hooks:
            response = hook(ctx, response)
        return response
