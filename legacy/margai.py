import time
import httpx
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI(title="marg.AI Namespace Proxy")

# Target upstream gateway (e.g., OpenAI, OpenRouter, vLLM, or Ollama)
UPSTREAM_BASE_URL = "https://api.openai.com/v1"
UPSTREAM_API_KEY = "sk-..."

# ==============================================================================
# 1. Dynamic Model Discovery (/v1/models)
# ==============================================================================

@app.get("/v1/models")
async def list_models(request: Request):
    """
    Fetches available models from the upstream provider and mirrors them
    with a 'marg/' prefix so they automatically populate in user dropdowns.
    """
    client = httpx.AsyncClient()
    headers = {"authorization": request.headers.get("authorization", f"Bearer {UPSTREAM_API_KEY}")}

    try:
        upstream_resp = await client.get(f"{UPSTREAM_BASE_URL}/models", headers=headers, timeout=10.0)
        upstream_data = upstream_resp.json()
    except Exception:
        # Fallback list if upstream isn't reachable
        upstream_data = {"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]}

    enhanced_models = []

    # 1. Expose the raw models as 'marg/<model-id>'
    for model in upstream_data.get("data", []):
        raw_id = model["id"]
        enhanced_models.append({
            "id": f"marg/{raw_id}",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "marg.ai",
            "parent": raw_id
        })

    # 2. Also keep the un-prefixed models available if users want standard pass-through
    enhanced_models.extend(upstream_data.get("data", []))

    return {"object": "list", "data": enhanced_models}


# ==============================================================================
# 2. Model Namespace Routing & Request Enhancement
# ==============================================================================

def process_model_and_prompt(body: Dict[str, Any]) -> tuple[bool, str, List[Dict[str, Any]]]:
    """
    Determines if the request should be enhanced based on model prefix OR bangtags.
    Strips the 'marg/' prefix so the upstream model gets its valid name.
    """
    req_model = body.get("model", "")
    messages = body.get("messages", [])

    is_marg_enhanced = False
    target_model = req_model

    # Check 1: Is user calling a 'marg/' namespaced model?
    if req_model.startswith("marg/"):
        is_marg_enhanced = True
        target_model = req_model.replace("marg/", "", 1) # Strip 'marg/' prefix

    # Check 2: Inspect messages for !marg bangtags
    updated_messages = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str) and "!marg" in content:
            is_marg_enhanced = True
            cleaned_content = content.replace("!marg", "").strip()
            updated_messages.append({
                "role": msg["role"],
                "content": f"{cleaned_content}\n\n[Context: Enhanced by marg.AI processing engine.]"
            })
        else:
            updated_messages.append(msg)

    # Incur system prompt modifications if enhanced
    if is_marg_enhanced:
        updated_messages.insert(0, {
            "role": "system",
            "content": "You are operating under the marg.AI runtime wrapper. Provide highly structured answers."
        })

    return is_marg_enhanced, target_model, updated_messages


# ==============================================================================
# 3. Main Chat Completion Endpoint
# ==============================================================================

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    # 1. Resolve model namespacing and prompt enhancements
    is_enhanced, target_model, updated_messages = process_model_and_prompt(body)

    # Update request payload for upstream
    body["model"] = target_model
    body["messages"] = updated_messages

    # 2. Forward request upstream
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    if "authorization" not in headers and UPSTREAM_API_KEY:
        headers["authorization"] = f"Bearer {UPSTREAM_API_KEY}"

    client = httpx.AsyncClient(timeout=60.0)
    req = client.build_request(
        method="POST",
        url=f"{UPSTREAM_BASE_URL}/chat/completions",
        headers=headers,
        json=body
    )

    upstream_response = await client.send(req, stream=body.get("stream", False))

    # 3. Handle Streaming Response
    if body.get("stream", False):
        async def stream_generator():
            async for line in upstream_response.aiter_lines():
                if line.startswith("data: [DONE]") and is_enhanced:
                    # Inject signature chunk before closing SSE stream
                    chunk = {
                        "id": "chatcmpl-marg",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": target_model,
                        "choices": [{"index": 0, "delta": {"content": "\n\nNote: enhanced with marg.AI"}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                elif line.strip():
                    yield f"{line}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # 4. Handle Non-Streaming Response
    res_json = upstream_response.json()
    if is_enhanced and "choices" in res_json and len(res_json["choices"]) > 0:
        res_json["choices"][0]["message"]["content"] += "\n\nNote: enhanced with marg.AI"

    return JSONResponse(content=res_json, status_code=upstream_response.status_code)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
