# Tomorrow's Strategy: Auto-Translator Proxy
@proxy.request_transformer
async def translate_to_english(ctx: RequestContext):
    # Tomorrow's custom logic: Inspect prompt, detect language, translate to English
    messages = ctx.body.get("messages", [])
    # ... translate messages ...
    ctx.state["original_language"] = "spanish"

@proxy.response_json_transformer
async def translate_back_to_spanish(response_json: dict, ctx: RequestContext):
    if ctx.state.get("original_language") == "spanish":
        # ... translate LLM response back to Spanish ...
        pass
    return response_json
