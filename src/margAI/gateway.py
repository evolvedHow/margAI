"""Gateway - Simplified API for margAI.

The Gateway provides a CrewAI-style decorator-first API where users define
bangtags as simple decorated functions. No magic, no hidden tags, complete
flexibility.

Example::

    from margAI import Gateway

    app = Gateway()

    @app.tag("rag")
    def add_context(ctx, value):
        docs = vector_db.search(value)
        ctx.add_system(f"Context: {docs}")

    @app.tag("fast")
    def route_fast(ctx):
        return ctx.route_to("local/llama3.2-1b")

    app.serve()

The framework owns the pipeline, exit points, and telemetry. You own the logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .bangtag import install_bangtags
from .config import load_config
from .wrapper import Wrapper

if TYPE_CHECKING:
    from .config import Config

__all__ = ["Gateway"]


class Gateway:
    """Simplified gateway API with decorator-first design.
    
    The Gateway wraps the full Wrapper API to provide a simpler, more
    opinionated interface focused on the most common use case: defining
    bangtags as decorated functions.
    
    Attributes:
        wrapper: The underlying Wrapper instance for advanced usage
    """

    def __init__(
        self,
        config: str | Path | Config | None = None,
        *,
        load_builtin_tags: bool = False,
    ) -> None:
        """Create a gateway.
        
        Args:
            config: Path to TOML config file, Config instance, or None to load
                from ./margAI.toml or $MARGAI_CONFIG
            load_builtin_tags: Whether to install built-in routing tags
                (route=, provider=, model=, etc.). Default False - define your own.
        
        Example::
        
            # Auto-load from ./margAI.toml
            app = Gateway()
            
            # Explicit config path
            app = Gateway("my_config.toml")
            
            # With built-in tags
            app = Gateway(load_builtin_tags=True)
        """
        # Load config
        if isinstance(config, (str, Path)):
            config_obj = load_config(config)
        elif config is None:
            config_obj = load_config()
        else:
            config_obj = config
        
        # Create wrapper (without auto-loading hooks)
        self.wrapper = Wrapper.from_config(config_obj, load_hooks=False)
        self._namespace = config_obj.gateway.prefix
        
        # Auto-install bangtag parsing (but no built-in tags unless requested)
        install_bangtags(self.wrapper, namespace=self._namespace)
        
        # Optionally install built-in tags
        if load_builtin_tags:
            from .builtin_tags import install as install_builtin_tags
            install_builtin_tags(self.wrapper)
    
    def tag(self, name: str, **kwargs: Any) -> Callable:
        """Register a bangtag handler.
        
        The decorated function is called when the bangtag appears in a prompt.
        
        Args:
            name: Tag name (without namespace prefix)
            **kwargs: Additional options (currently unused, reserved for future)
        
        Returns:
            Decorator function
        
        Usage::
        
            @app.tag("summarize")
            def summarize(ctx, tag):
                ctx.add_system("Provide a brief summary.")
            
            @app.tag("lookup")
            def lookup(ctx, tag):
                query = tag.value or ctx.user_message()
                docs = search(query)
                ctx.add_system(f"Context: {docs}")
        
        The tag function can:
        
        - Modify the prompt: ``ctx.add_system(...)``, ``ctx.set_user_message(...)``
        - Override routing: ``return ctx.route_to("provider/model")``
        - Short-circuit: ``return ctx.respond({...})``
        - Return error: ``return ctx.error(400, "message")``
        - Continue normally: Don't return anything (or return None)
        """
        def decorator(func: Callable) -> Callable:
            # Register as a before hook (tag-keyed)
            self.wrapper.before(name)(func)
            return func
        return decorator
    
    def on_request(self, func: Callable) -> Callable:
        """Register a hook that runs on every request.
        
        Args:
            func: Function that takes ctx and modifies it
        
        Returns:
            The same function (for chaining)
        
        Usage::
        
            @app.on_request
            def log_request(ctx):
                print(f"Request to {ctx.body.get('model')}")
        """
        return self.wrapper.before(func)
    
    def on_response(self, func: Callable) -> Callable:
        """Register a hook that runs on every response.
        
        Args:
            func: Function that takes (payload, ctx) and returns modified payload
        
        Returns:
            The same function (for chaining)
        
        Usage::
        
            @app.on_response
            def log_response(payload, ctx):
                tokens = payload.get('usage', {}).get('total_tokens', 0)
                print(f"Response: {tokens} tokens")
                return payload
        """
        return self.wrapper.after(func)
    
    def on_stream(self, func: Callable) -> Callable:
        """Register a hook that runs on each streaming chunk.
        
        Args:
            func: Function that takes (chunk, ctx) and returns modified chunk
        
        Returns:
            The same function (for chaining)
        
        Usage::
        
            @app.on_stream
            def process_chunk(chunk, ctx):
                # Modify or filter chunks
                return chunk
        """
        return self.wrapper.stream(func)
    
    def on_error(self, func: Callable) -> Callable:
        """Register a hook that runs on errors.
        
        Args:
            func: Function that takes (exc, ctx) and optionally returns error dict
        
        Returns:
            The same function (for chaining)
        
        Usage::
        
            @app.on_error
            def handle_error(exc, ctx):
                # Log error, alert, etc.
                if isinstance(exc, TimeoutError):
                    return {"error": {"message": "Request timed out"}}
        """
        return self.wrapper.error(func)
    
    def serve(
        self,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        """Start the gateway server.
        
        Args:
            host: Host to bind to (default from config)
            port: Port to bind to (default from config)
        
        Usage::
        
            app = Gateway()
            
            @app.tag("summarize")
            def summarize(ctx, tag):
                ctx.add_system("Be concise.")
            
            app.serve()  # Starts on config host:port
        """
        import uvicorn
        from .transport.fastapi import build_app
        
        config = self.wrapper.config
        uvicorn.run(
            build_app(wrapper=self.wrapper),
            host=host or config.gateway.host,
            port=port or config.gateway.port,
        )
    
    @property
    def namespace(self) -> str:
        """The bangtag namespace (from config gateway.prefix).
        
        This is what users write in prompts: !{namespace}: <tags>
        """
        return self._namespace
    
    def describe(self) -> dict[str, Any]:
        """Describe the gateway: tags, models, config, etc.
        
        Returns:
            Dict with registered tags, models, providers, etc.
        
        Usage::
        
            app = Gateway()
            
            @app.tag("summarize")
            def summarize(ctx, tag): ...
            
            info = app.describe()
            print(info["tags"])  # Lists registered tags
        """
        return self.wrapper.describe()
