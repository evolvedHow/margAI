"""Main proxy class - ties everything together."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

try:
    from .router import Router, Backend
    from .hooks import HookManager, HookContext, BeforeHook, AfterHook
    from .transport import Transport
except ImportError:
    from router import Router, Backend
    from hooks import HookManager, HookContext, BeforeHook, AfterHook
    from transport import Transport

logger = logging.getLogger("marga")


class Proxy:
    """Ultra-minimal LLM routing proxy.
    
    A transparent passthrough layer with smart routing and hooks.
    
    Example:
        proxy = Proxy(backends={
            "openai": {"base_url": "https://api.openai.com/v1", "api_key": "sk-..."},
            "local": {"base_url": "http://127.0.0.1:8080/v1"}
        })
        
        @proxy.route
        def smart_route(ctx):
            return "local/qwen2.5-coder" if ctx.estimated_tokens < 1000 else "openai/gpt-4o-mini"
        
        @proxy.before
        def add_context(ctx):
            ctx.messages.insert(0, {"role": "system", "content": "..."})
        
        proxy.serve()
    """
    
    def __init__(self, backends: dict[str, dict[str, Any]]):
        """Initialize proxy with backend configurations.
        
        Args:
            backends: Dict of backend configs, e.g.:
                {
                    "openai": {
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "sk-...",
                        "models": ["gpt-4o", "gpt-4o-mini"]
                    }
                }
        """
        # Convert dict configs to Backend objects
        backend_objs = {
            name: Backend(
                name=name,
                base_url=cfg["base_url"],
                api_key=cfg.get("api_key", ""),
                models=cfg.get("models", [])
            )
            for name, cfg in backends.items()
        }
        
        self.router = Router(backend_objs)
        self.hooks = HookManager()
        self.transport = Transport()
        self._app: FastAPI | None = None
    
    def route(self, fn):
        """Decorator to register custom routing function.
        
        Example:
            @proxy.route
            def my_router(ctx: HookContext) -> str:
                return "backend/model"
        """
        self.router.set_router(fn)
        return fn
    
    def before(self, fn: BeforeHook):
        """Decorator to register before hook.
        
        Example:
            @proxy.before
            def inject_rag(ctx: HookContext):
                ctx.messages.insert(0, {...})
        """
        self.hooks.add_before(fn)
        return fn
    
    def after(self, fn: AfterHook):
        """Decorator to register after hook.
        
        Example:
            @proxy.after
            def log_tokens(ctx: HookContext, response: dict) -> dict:
                print(f"Used {response['usage']['total_tokens']} tokens")
                return response
        """
        self.hooks.add_after(fn)
        return fn
    
    def build_app(self) -> FastAPI:
        """Build FastAPI application."""
        app = FastAPI(title="marga")
        
        @app.post("/v1/chat/completions")
        async def chat_completions(body: dict[str, Any]):
            """OpenAI-compatible chat completions endpoint."""
            try:
                # Extract request details
                messages = body.get("messages", [])
                model = body.get("model", "")
                stream = body.get("stream", False)
                
                # Create hook context
                ctx = HookContext(
                    messages=messages.copy(),
                    model=model,
                    stream=stream
                )
                
                # Run before hooks (can modify ctx)
                self.hooks.run_before(ctx)
                
                # Route to backend
                backend, resolved_model = self.router.route(ctx)
                
                # Update body with potentially modified messages and model
                body["messages"] = ctx.messages
                body["model"] = resolved_model
                
                # Handle streaming vs non-streaming
                if stream:
                    async def stream_response():
                        async for chunk in self.transport.stream(
                            backend.base_url,
                            backend.api_key,
                            body
                        ):
                            yield chunk
                    
                    return StreamingResponse(
                        stream_response(),
                        media_type="text/event-stream"
                    )
                else:
                    # Complete request
                    response = await self.transport.complete(
                        backend.base_url,
                        backend.api_key,
                        body
                    )
                    
                    # Run after hooks
                    response = self.hooks.run_after(ctx, response)
                    
                    # Record usage
                    tokens = response.get("usage", {}).get("total_tokens", 0)
                    self.router.record_usage(backend.name, tokens)
                    
                    return response
                    
            except Exception as e:
                logger.error(f"Proxy error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/v1/models")
        async def list_models():
            """List available models from all backends."""
            models = []
            for backend in self.router.backends.values():
                for model in backend.models:
                    models.append({
                        "id": f"{backend.name}/{model}",
                        "object": "model",
                        "owned_by": backend.name
                    })
            return {"object": "list", "data": models}
        
        @app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "ok"}
        
        @app.get("/stats")
        async def stats():
            """Get routing statistics."""
            return self.router.get_stats()
        
        self._app = app
        return app
    
    def serve(self, host: str = "127.0.0.1", port: int = 8000):
        """Start the proxy server.
        
        Args:
            host: Host to bind to (default: 127.0.0.1 for security)
            port: Port to bind to
        """
        app = self.build_app()
        uvicorn.run(app, host=host, port=port)
