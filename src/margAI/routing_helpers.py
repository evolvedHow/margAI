"""Optional routing helpers for common patterns.

These are convenience functions you can use with @app.tag() to handle
common routing scenarios. They're optional - you can always write your own.

Example::

    from margAI import Gateway
    from margAI.routing_helpers import cheapest, fastest
    
    app = Gateway()
    
    # Use built-in helper
    app.tag("cheap")(cheapest)
    
    # Or write your own
    @app.tag("smart")
    def smart_route(ctx):
        if ctx.analyze_complexity() == "high":
            return ctx.route_to("openai/gpt-4o")
        else:
            return ctx.route_to("local/llama")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core.context import RequestContext
    from .core.responses import RouteOverride

__all__ = [
    "cheapest",
    "fastest",
    "smart_complexity",
    "pass_through",
]


def cheapest(ctx: RequestContext, tag: any = None) -> RouteOverride | None:
    """Route to the cheapest available model.
    
    Requires costs configured in [telemetry.costs].
    
    Example::
    
        from margAI import Gateway
        from margAI.routing_helpers import cheapest
        
        app = Gateway()
        app.tag("cheap")(cheapest)
        
        # User writes: !app: cheap What is Python?
    
    Returns:
        RouteOverride to cheapest model, or None if no costs configured
    """
    # TODO: Implement once ModelCollection is ready
    # For now, this is a placeholder showing the pattern
    return None


def fastest(ctx: RequestContext, tag: any = None) -> RouteOverride | None:
    """Route to the fastest model (by historical latency).
    
    Uses telemetry data to find the model with lowest average latency.
    Falls back to first available model if no latency data exists.
    
    Example::
    
        from margAI import Gateway
        from margAI.routing_helpers import fastest
        
        app = Gateway()
        app.tag("fast")(fastest)
    
    Returns:
        RouteOverride to fastest model
    """
    # TODO: Implement once ModelCollection is ready
    return None


def smart_complexity(ctx: RequestContext, tag: any = None) -> RouteOverride | None:
    """Route based on query complexity.
    
    Uses ctx.analyze_complexity() to route:
    - Low complexity: Cheapest model
    - Medium complexity: Balanced model
    - High complexity: Best model
    
    Example::
    
        from margAI import Gateway
        from margAI.routing_helpers import smart_complexity
        
        app = Gateway()
        app.tag("smart")(smart_complexity)
    
    Returns:
        RouteOverride based on complexity
    """
    complexity = ctx.analyze_complexity()
    
    # TODO: Implement proper model selection once ModelCollection is ready
    # For now, show the pattern
    
    if complexity == "low":
        # Route to cheapest
        return None
    elif complexity == "high":
        # Route to best
        return None
    else:
        # Route to balanced
        return None


def pass_through(ctx: RequestContext, tag: any = None) -> None:
    """Do nothing - useful for documentation/testing.
    
    Example::
    
        from margAI import Gateway
        from margAI.routing_helpers import pass_through
        
        app = Gateway()
        app.tag("noop")(pass_through)
    """
    pass
