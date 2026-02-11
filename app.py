import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from aiohttp import WSMsgType, web
import edge_tts


BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR / "frontend"
MAX_SEGMENT_CHARS = 3000
HISTORY_DIR = BASE_DIR / "data" / "history"
HISTORY_FILE = HISTORY_DIR / "records.json"
MAX_HISTORY_ITEMS = 30


class SynthesisStoppedError(Exception):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_history_storage() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text("[]", encoding="utf-8")


def load_history_records() -> list[dict[str, Any]]:
    ensure_history_storage()
    try:
        records = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(records, list):
        return []

    return [record for record in records if isinstance(record, dict)]


def save_history_records(records: list[dict[str, Any]]) -> None:
    ensure_history_storage()
    HISTORY_FILE.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def trim_history(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stale_records = records[MAX_HISTORY_ITEMS:]
    for stale_record in stale_records:
        stale_filename = stale_record.get("audio_filename")
        if stale_filename:
            stale_audio_path = HISTORY_DIR / stale_filename
            if stale_audio_path.exists():
                stale_audio_path.unlink()
    return records[:MAX_HISTORY_ITEMS]


def get_history_record(record_id: str) -> dict[str, Any] | None:
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


async def synthesize_speech_iter(
    text: str,
    voice: str,
    rate: str,
    pitch: str,
    stop_event: asyncio.Event,
) -> bytes:
    audio_chunks: list[bytes] = []

    for text_segment in split_text_segments(text):
        if stop_event.is_set():
            raise SynthesisStoppedError("Synthesis stopped by user")

        communicate = edge_tts.Communicate(
            text=text_segment,
            voice=voice,
            rate=rate,
            pitch=pitch,
        )

        async for chunk in communicate.stream():
            if stop_event.is_set():
                raise SynthesisStoppedError("Synthesis stopped by user")

            if chunk.get("type") == "audio":
                audio_chunks.append(chunk["data"])

    return b"".join(audio_chunks)


class SynthesisManager:
    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._stop_events: dict[str, asyncio.Event] = {}
        self._history_lock = asyncio.Lock()

    async def _update_history(self, record_id: str, updates: dict[str, Any]) -> None:
        async with self._history_lock:
            records = load_history_records()
            for record in records:
                if record.get("id") == record_id:
                    record.update(updates)
                    break
            save_history_records(trim_history(records))

    async def _create_history_record(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._history_lock:
            records = load_history_records()
            records.insert(0, payload)
            save_history_records(trim_history(records))
        return payload

    async def start(self, text: str, voice: str, rate: str, pitch: str) -> dict[str, Any]:
        record_id = uuid4().hex
        stop_event = asyncio.Event()
        self._stop_events[record_id] = stop_event

        record = {
            "id": record_id,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "text": text,
            "voice": voice,
            "rate": rate,
            "pitch": pitch,
            "status": "in_progress",
            "audio_filename": None,
            "error": None,
        }
        await self._create_history_record(record)

        self._tasks[record_id] = asyncio.create_task(
            self._run_synthesis(
                record_id=record_id,
                text=text,
                voice=voice,
                rate=rate,
                pitch=pitch,
                stop_event=stop_event,
            )
        )
        return record

    async def _run_synthesis(
        self,
        record_id: str,
        text: str,
        voice: str,
        rate: str,
        pitch: str,
        stop_event: asyncio.Event,
    ) -> None:
        audio_data = b""
        try:
            audio_data = await synthesize_speech_iter(text, voice, rate, pitch, stop_event)

            filename = None
            if audio_data:
                filename = f"{record_id}.mp3"
                (HISTORY_DIR / filename).write_bytes(audio_data)

            await self._update_history(
                record_id,
                {
                    "status": "completed",
                    "audio_filename": filename,
                    "updated_at": now_iso(),
                    "completed_at": now_iso(),
                },
            )
        except SynthesisStoppedError:
            filename = None
            if audio_data:
                filename = f"{record_id}.mp3"
                (HISTORY_DIR / filename).write_bytes(audio_data)

            await self._update_history(
                record_id,
                {
                    "status": "stopped",
                    "audio_filename": filename,
                    "updated_at": now_iso(),
                    "stopped_at": now_iso(),
                },
            )
        except Exception as exc:
            await self._update_history(
                record_id,
                {
                    "status": "failed",
                    "error": str(exc),
                    "updated_at": now_iso(),
                },
            )
        finally:
            self._tasks.pop(record_id, None)
            self._stop_events.pop(record_id, None)

    async def stop(self, record_id: str) -> bool:
        stop_event = self._stop_events.get(record_id)
        if not stop_event:
            return False

        stop_event.set()
        await self._update_history(record_id, {"status": "stopping", "updated_at": now_iso()})
        return True

    def is_running(self, record_id: str) -> bool:
        task = self._tasks.get(record_id)
        return task is not None and not task.done()


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)

    await ws.send_json(
        {
            "type": "ready",
            "message": "Connected. Use /api/synthesis for resilient background synthesis.",
        }
    )

    manager: SynthesisManager = request.app["synthesis_manager"]

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
        if action == "synthesize":
            text = (payload.get("text") or "").strip()
            voice = payload.get("voice") or "en-US-JennyNeural"
            rate = payload.get("rate") or "+0%"
            pitch = payload.get("pitch") or "+0Hz"

            if not text:
                await ws.send_json({"type": "error", "message": "Text is required"})
                continue

            record = await manager.start(text=text, voice=voice, rate=rate, pitch=pitch)
            await ws.send_json(
                {
                    "type": "status",
                    "message": "Synthesis started in background",
                    "record_id": record["id"],
                }
            )
            continue

        if action == "stop":
            record_id = (payload.get("record_id") or "").strip()
            if not record_id:
                await ws.send_json({"type": "error", "message": "record_id is required"})
                continue
            did_stop = await manager.stop(record_id)
            if not did_stop:
                await ws.send_json({"type": "error", "message": "No running synthesis for record_id"})
                continue
            await ws.send_json({"type": "status", "message": "Stop requested", "record_id": record_id})
            continue

        await ws.send_json({"type": "error", "message": "Unsupported action"})

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


async def synthesize_handler(request: web.Request) -> web.Response:
    manager: SynthesisManager = request.app["synthesis_manager"]

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        raise web.HTTPBadRequest(reason="Invalid JSON payload")

    text = (payload.get("text") or "").strip()
    voice = payload.get("voice") or "en-US-JennyNeural"
    rate = payload.get("rate") or "+0%"
    pitch = payload.get("pitch") or "+0Hz"

    if not text:
        raise web.HTTPBadRequest(reason="Text is required")

    record = await manager.start(text=text, voice=voice, rate=rate, pitch=pitch)
    return web.json_response({"item": record}, status=202)


async def stop_history_handler(request: web.Request) -> web.Response:
    manager: SynthesisManager = request.app["synthesis_manager"]
    record_id = request.match_info.get("record_id", "")

    if not record_id:
        raise web.HTTPBadRequest(reason="record_id is required")

    did_stop = await manager.stop(record_id)
    if not did_stop:
        raise web.HTTPNotFound(reason="No running synthesis with this record_id")

    return web.json_response({"ok": True, "record_id": record_id})


def create_app(serve_frontend: bool = True) -> web.Application:
    app = web.Application()
    app["synthesis_manager"] = SynthesisManager()

    if serve_frontend:
        app.router.add_get("/", index_handler)
        app.router.add_static("/frontend", FRONTEND_DIR)

    app.router.add_get("/health", health_handler)
    app.router.add_get("/api/history", history_handler)
    app.router.add_get("/api/history/{record_id}/audio", history_audio_handler)
    app.router.add_post("/api/synthesis", synthesize_handler)
    app.router.add_post("/api/history/{record_id}/stop", stop_history_handler)
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
