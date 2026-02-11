import base64
import json
from pathlib import Path

from aiohttp import WSMsgType, web
import edge_tts


BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"


async def synthesize_speech(text: str, voice: str = "en-US-JennyNeural") -> bytes:
    communicate = edge_tts.Communicate(text=text, voice=voice)
    audio_chunks: list[bytes] = []

    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            audio_chunks.append(chunk["data"])

    return b"".join(audio_chunks)


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)

    await ws.send_json({"type": "ready", "message": "Connected to Edge TTS server"})

    async for msg in ws:
        if msg.type != WSMsgType.TEXT:
            if msg.type == WSMsgType.ERROR:
                print(f"WebSocket closed with exception: {ws.exception()}")
            continue

        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            await ws.send_json({"type": "error", "message": "Invalid JSON payload"})
            continue

        action = payload.get("action")
        if action != "synthesize":
            await ws.send_json({"type": "error", "message": "Unsupported action"})
            continue

        text = (payload.get("text") or "").strip()
        voice = payload.get("voice") or "en-US-JennyNeural"

        if not text:
            await ws.send_json({"type": "error", "message": "Text is required"})
            continue

        if len(text) > 3000:
            await ws.send_json({"type": "error", "message": "Text too long (max 3000 characters)"})
            continue

        await ws.send_json({"type": "status", "message": "Synthesizing..."})

        try:
            audio_data = await synthesize_speech(text=text, voice=voice)
            audio_b64 = base64.b64encode(audio_data).decode("utf-8")
            await ws.send_json(
                {
                    "type": "audio",
                    "format": "audio/mpeg",
                    "filename": "speech.mp3",
                    "audio_base64": audio_b64,
                }
            )
        except Exception as exc:
            await ws.send_json({"type": "error", "message": f"Synthesis failed: {exc}"})

    return ws


async def index_handler(_: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "index.html")


async def health_handler(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/health", health_handler)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/static", STATIC_DIR)
    return app


if __name__ == "__main__":
    web.run_app(create_app(), host="0.0.0.0", port=8080)
