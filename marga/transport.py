"""HTTP transport for backend communication."""

from __future__ import annotations

import httpx
from typing import Any, AsyncIterator


class Transport:
    """Handles HTTP communication with LLM backends."""
    
    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def complete(
        self,
        base_url: str,
        api_key: str,
        body: dict[str, Any]
    ) -> dict[str, Any]:
        """Send completion request to backend.
        
        Args:
            base_url: Backend base URL (e.g., "https://api.openai.com/v1")
            api_key: API key for authentication
            body: OpenAI-format request body
            
        Returns:
            OpenAI-format response
        """
        client = await self._get_client()
        
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        url = f"{base_url.rstrip('/')}/chat/completions"
        
        response = await client.post(
            url,
            json=body,
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    
    async def stream(
        self,
        base_url: str,
        api_key: str,
        body: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Stream completion from backend.
        
        Args:
            base_url: Backend base URL
            api_key: API key for authentication
            body: OpenAI-format request body (with stream=True)
            
        Yields:
            SSE chunks from backend
        """
        client = await self._get_client()
        
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        url = f"{base_url.rstrip('/')}/chat/completions"
        
        async with client.stream(
            "POST",
            url,
            json=body,
            headers=headers
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                yield chunk
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
