# proxy.py — AI Proxy Server for Groq + Edge TTS (merged & hardened)
import os
import httpx
from fastapi import FastAPI, HTTPException, Header, Request, Response
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
import edge_tts

# ============ CONFIG FROM ENV ============
PROXY_ID      = os.getenv("PROXY_ID", "proxy_unknown")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
PROXY_SECRET  = os.getenv("PROXY_SECRET", "")

# ============ MODELS (HARDCODED) ============
GROQ_TEXT_MODEL   = "openai/gpt-oss-120b"
GROQ_VISION_MODEL = "qwen/qwen3.8-27b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ============ HTTP CLIENT ============
http_client: httpx.AsyncClient = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=60.0)
    print(f"[{PROXY_ID}] started | text={GROQ_TEXT_MODEL} | vision={GROQ_VISION_MODEL} | groq_key={'SET' if GROQ_API_KEY else 'MISSING'}")
    yield
    await http_client.aclose()
    print(f"[{PROXY_ID}] stopped")

app = FastAPI(title=f"AI Proxy {PROXY_ID}", lifespan=lifespan)

def check_auth(x_proxy_key: str):
    """Shared auth — ব্যবহার করুন সব endpoint-এ"""
    if not PROXY_SECRET or x_proxy_key != PROXY_SECRET:
        raise HTTPException(status_code=401, detail="Invalid proxy key")

def has_image_content(messages) -> bool:
    if not isinstance(messages, list):
        return False
    for msg in messages:
        if isinstance(msg, dict) and isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False

# ============ GROQ CHAT ENDPOINT ============
@app.post("/chat/completions")
async def proxy_chat(request: Request, x_proxy_key: str = Header(None)):
    check_auth(x_proxy_key)
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Proxy not configured")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if has_image_content(body.get("messages")):
        body["model"] = GROQ_VISION_MODEL
        body.setdefault("reasoning_effort", "low")
        print(f"[{PROXY_ID}] → VISION: {GROQ_VISION_MODEL}")
    else:
        body["model"] = GROQ_TEXT_MODEL
        print(f"[{PROXY_ID}] → TEXT: {GROQ_TEXT_MODEL}")

    try:
        response = await http_client.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json=body
        )
        if response.status_code >= 400:
            return JSONResponse(status_code=response.status_code, content=response.json())
        return response.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail=f"[{PROXY_ID}] Groq timeout")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"[{PROXY_ID}] Proxy error: {str(e)}")

# ============ EDGE TTS ENDPOINT (auth-protected) ============
class TTSRequest(BaseModel):
    text: str
    voice: str = "bn-BD-PradeepNeural"   # ডিফল্ট বাংলা ভয়েস
    rate: str = "+0%"                    # যেমন "+10%" / "-20%"
    volume: str = "+0%"

@app.post("/tts")
async def generate_tts(req: TTSRequest, x_proxy_key: str = Header(None)):
    check_auth(x_proxy_key)              # 🔒 এখন auth লাগবে
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        print(f"[{PROXY_ID}] → TTS: voice={req.voice} len={len(req.text)}")
        communicate = edge_tts.Communicate(req.text, req.voice, rate=req.rate, volume=req.volume)

        # ⏱️ timeout wrapper — hang হলে 55 সেকেন্ডে fail করবে
        audio_data = b""
        async def _collect():
            nonlocal audio_data
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

        await asyncio.wait_for(_collect(), timeout=55.0)

        if not audio_data:
            raise HTTPException(status_code=502, detail="TTS returned no audio")

        return Response(content=audio_data, media_type="audio/mpeg")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"[{PROXY_ID}] TTS timeout")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

# ============ HEALTH & INFO (full version restored) ============
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "groq+tts",
        "proxy_id": PROXY_ID,
        "text_model": GROQ_TEXT_MODEL,
        "vision_model": GROQ_VISION_MODEL,
        "groq_configured": bool(GROQ_API_KEY),
        "secret_configured": bool(PROXY_SECRET),
        "tts_enabled": True,
    }

@app.get("/")
async def root():
    return {"service": "AI Proxy", "proxy_id": PROXY_ID, "status": "running"}

# ✅ /my-ip restored
@app.get("/my-ip")
async def get_my_ip():
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get("https://api.ipify.org?format=json")
            return {"proxy_id": PROXY_ID, **r.json()}
    except Exception as e:
        return {"proxy_id": PROXY_ID, "error": str(e)}

# ============ RUN ============
if __name__ == "__main__":
    import asyncio
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
