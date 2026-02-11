import base64
import argparse
from datetime import datetime, timezone
import json
from uuid import uuid4
from pathlib import Path

from aiohttp import WSMsgType, web
import edge_tts


BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR / "frontend"
MAX_SEGMENT_CHARS = 3000
HISTORY_DIR = BASE_DIR / "data" / "history"
HISTORY_FILE = HISTORY_DIR / "records.json"
MAX_HISTORY_ITEMS = 30


def ensure_history_storage() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text("[]", encoding="utf-8")


def load_history_records() -> list[dict]:
    ensure_history_storage()
    try:
        records = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(records, list):
        return []

    return [record for record in records if isinstance(record, dict)]


def save_history_records(records: list[dict]) -> None:
    ensure_history_storage()
    HISTORY_FILE.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_audio_history_entry(
    text: str,
    voice: str,
    rate: str,
    pitch: str,
    audio_data: bytes,
) -> dict:
    record_id = uuid4().hex
    filename = f"{record_id}.mp3"
    audio_path = HISTORY_DIR / filename
    audio_path.write_bytes(audio_data)

    records = load_history_records()
    record = {
        "id": record_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "voice": voice,
        "rate": rate,
        "pitch": pitch,
        "audio_filename": filename,
    }
    records.insert(0, record)

    stale_records = records[MAX_HISTORY_ITEMS:]
    for stale_record in stale_records:
        stale_filename = stale_record.get("audio_filename")
        if stale_filename:
            stale_audio_path = HISTORY_DIR / stale_filename
            if stale_audio_path.exists():
                stale_audio_path.unlink()

    save_history_records(records[:MAX_HISTORY_ITEMS])
    return record


def get_history_record(record_id: str) -> dict | None:
    records = load_history_records()
    for record in records:
        if record.get("id") == record_id:
            return record
    return None


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

            write_audio_history_entry(
                text=text,
                voice=voice,
                rate=rate,
                pitch=pitch,
                audio_data=audio_data,
            )
        except Exception as exc:
            await ws.send_json({"type": "error", "message": f"Synthesis failed: {exc}"})

    return ws


async def index_handler(_: web.Request) -> web.FileResponse:
    return web.FileResponse(FRONTEND_DIR / "index.html")


async def health_handler(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def history_handler(_: web.Request) -> web.Response:
    records = load_history_records()
    return web.json_response({"items": records})


async def history_audio_handler(request: web.Request) -> web.StreamResponse:
    record_id = request.match_info.get("record_id", "")
    record = get_history_record(record_id)
    if not record:
        raise web.HTTPNotFound(reason="History record not found")

    audio_filename = record.get("audio_filename")
    if not audio_filename:
        raise web.HTTPNotFound(reason="Audio file metadata missing")

    audio_path = HISTORY_DIR / audio_filename
    if not audio_path.exists():
        raise web.HTTPNotFound(reason="Audio file not found")

    return web.FileResponse(audio_path, headers={"Content-Type": "audio/mpeg"})


def create_app(serve_frontend: bool = True) -> web.Application:
    app = web.Application()

    if serve_frontend:
        app.router.add_get("/", index_handler)
        app.router.add_static("/frontend", FRONTEND_DIR)

    app.router.add_get("/health", health_handler)
    app.router.add_get("/api/history", history_handler)
    app.router.add_get("/api/history/{record_id}/audio", history_audio_handler)
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
