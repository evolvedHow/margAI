import time
import json
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(titlle="Marg.AI showing the way to the right model.")

# Pydantic models for Request validations

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str
    user: Optional[str] = None


# Helper generators for streaming server-send Events

async def stream_chat_completion_stub(model: str):
    """Generates SSE (Server-Sent Events) formatted for OpenAI chat streaming."""
    req_id = "chatcmpl-stub123"
    created_time = int(time.time())

    # 1. Send initial role payload
    yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

    # 2. Stream content chunks
    stub_chunks = ["This ", "is ", "a ", "stubbed ", "streaming ", "response."]
    for chunk in stub_chunks:
        # TODO: Replace this loop with your actual LLM token generator
        payload = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(payload)}\n\n"

    # 3. Send final payload with finish_reason
    yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"

    # 4. Terminate stream
    yield "data: [DONE]\n\n"

# API endpoints.  These mirror OpenAI APIs

@app.get("/v1/models")
async def list_models():
    """Lists available models."""
    # TODO: Dynamically return the models loaded in your wrapper
    return {
        "object": "list",
        "data": [
            {
                "id": "my-custom-model-v1",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "my-organization"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Primary endpoint for chat-based LLM interactions."""

    if request.stream:
        return StreamingResponse(
            stream_chat_completion_stub(request.model),
            media_type="text/event-stream"
        )

    # TODO: Integrate your actual LLM synchronous inference here
    response_content = "This is a stubbed synchronous chat response."

    return {
        "id": "chatcmpl-stub123",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20
        }
    }

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """Endpoint for generating text embeddings."""

    # TODO: Integrate your actual embedding model here
    # Mocking a 1536-dimensional embedding array
    dummy_embedding = [0.0] * 1536

    # Handle single string vs list of strings
    inputs = request.input if isinstance(request.input, list) else [request.input]

    data = []
    for i, _ in enumerate(inputs):
        data.append({
            "object": "embedding",
            "embedding": dummy_embedding,
            "index": i
        })

    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {
            "prompt_tokens": 8,
            "total_tokens": 8
        }
    }

if __name__ == "__main__":
    import uvicorn
    # Run via: python main.py OR uvicorn main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
