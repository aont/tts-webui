import base64
import argparse
import json
from pathlib import Path

from aiohttp import WSMsgType, web
import edge_tts


BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR / "frontend"
MAX_SEGMENT_CHARS = 3000


def split_text_segments(text: str, max_chars: int = MAX_SEGMENT_CHARS) -> list[str]:
    """Split text into chunks that respect maximum length while preferring word boundaries."""
    stripped_text = text.strip()
    if not stripped_text:
        return []

    words = stripped_text.split()
    segments: list[str] = []
    current_segment: list[str] = []
    current_length = 0

    for word in words:
        if len(word) > max_chars:
            if current_segment:
                segments.append(" ".join(current_segment))
                current_segment = []
                current_length = 0

            for i in range(0, len(word), max_chars):
                segments.append(word[i : i + max_chars])
            continue

        separator_length = 1 if current_segment else 0
        projected_length = current_length + separator_length + len(word)

        if projected_length <= max_chars:
            current_segment.append(word)
            current_length = projected_length
        else:
            segments.append(" ".join(current_segment))
            current_segment = [word]
            current_length = len(word)

    if current_segment:
        segments.append(" ".join(current_segment))

    return segments


async def synthesize_speech(
    text: str,
    voice: str = "en-US-JennyNeural",
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> bytes:
    audio_chunks: list[bytes] = []

    for text_segment in split_text_segments(text):
        communicate = edge_tts.Communicate(
            text=text_segment,
            voice=voice,
            rate=rate,
            pitch=pitch,
        )

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
        rate = payload.get("rate") or "+0%"
        pitch = payload.get("pitch") or "+0Hz"

        if not text:
            await ws.send_json({"type": "error", "message": "Text is required"})
            continue

        await ws.send_json({"type": "status", "message": "Synthesizing..."})

        try:
            audio_data = await synthesize_speech(
                text=text,
                voice=voice,
                rate=rate,
                pitch=pitch,
            )
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
    return web.FileResponse(FRONTEND_DIR / "index.html")


async def health_handler(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


def create_app(serve_frontend: bool = True) -> web.Application:
    app = web.Application()

    if serve_frontend:
        app.router.add_get("/", index_handler)
        app.router.add_static("/frontend", FRONTEND_DIR)

    app.router.add_get("/health", health_handler)
    app.router.add_get("/ws", ws_handler)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edge TTS WebSocket server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind (default: 8080)",
    )
    parser.add_argument(
        "--no-frontend",
        action="store_true",
        help="Disable serving frontend files from this process",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    web.run_app(
        create_app(serve_frontend=not args.no_frontend),
        host=args.host,
        port=args.port,
    )
