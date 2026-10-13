"""marga: Ultra-minimal LLM routing proxy.

Sanskrit: मार्ग (mārga) - "path" or "route"

A transparent passthrough layer with smart routing and prompt hooks.
"""

from .proxy import Proxy
from .router import Router, Backend
from .hooks import HookContext

__version__ = "0.1.0"

__all__ = ["Proxy", "Router", "Backend", "HookContext"]
