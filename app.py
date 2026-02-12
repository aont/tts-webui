import argparse
import asyncio
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any
from uuid import uuid4
import wave

from aiohttp import ClientError, ClientSession, web
import edge_tts


BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR / "frontend"
MAX_SEGMENT_CHARS = 3000
HISTORY_DIR = BASE_DIR / "data" / "history"
HISTORY_FILE = HISTORY_DIR / "records.json"
MAX_HISTORY_ITEMS = 30
DEFAULT_ENGINE = "edge-tts"
DEFAULT_VOICEVOX_ENGINE_URL = "http://127.0.0.1:50021"


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


async def synthesize_speech_voicevox(
    text: str,
    speaker: int,
    voicevox_engine_url: str,
    stop_event: asyncio.Event,
) -> bytes:
    if stop_event.is_set():
        raise SynthesisStoppedError("Synthesis stopped by user")

    wav_chunks: list[bytes] = []

    async with ClientSession() as session:
        for text_segment in split_text_segments(text):
            if stop_event.is_set():
                raise SynthesisStoppedError("Synthesis stopped by user")

            query_url = f"{voicevox_engine_url}/audio_query"
            synth_url = f"{voicevox_engine_url}/synthesis"

            async with session.post(
                query_url,
                params={"text": text_segment, "speaker": speaker},
            ) as query_response:
                if query_response.status >= 400:
                    error_body = await query_response.text()
                    raise RuntimeError(
                        f"VoiceVox audio_query failed ({query_response.status}): {error_body}"
                    )
                audio_query = await query_response.json()

            async with session.post(
                synth_url,
                params={"speaker": speaker},
                json=audio_query,
            ) as synth_response:
                if synth_response.status >= 400:
                    error_body = await synth_response.text()
                    raise RuntimeError(
                        f"VoiceVox synthesis failed ({synth_response.status}): {error_body}"
                    )
                wav_chunks.append(await synth_response.read())

    return encode_voicevox_wav_chunks_to_opus(wav_chunks)


def encode_voicevox_wav_chunks_to_opus(wav_chunks: list[bytes]) -> bytes:
    if not wav_chunks:
        return b""

    channels: int | None = None
    sample_width: int | None = None
    frame_rate: int | None = None
    pcm_frames: list[bytes] = []

    for chunk in wav_chunks:
        with wave.open(io.BytesIO(chunk), "rb") as wav_reader:
            chunk_channels = wav_reader.getnchannels()
            chunk_sample_width = wav_reader.getsampwidth()
            chunk_frame_rate = wav_reader.getframerate()

            if channels is None:
                channels = chunk_channels
                sample_width = chunk_sample_width
                frame_rate = chunk_frame_rate
            elif (
                chunk_channels != channels
                or chunk_sample_width != sample_width
                or chunk_frame_rate != frame_rate
            ):
                raise RuntimeError("VoiceVox returned WAV chunks with mismatched audio format")

            pcm_frames.append(wav_reader.readframes(wav_reader.getnframes()))

    if channels is None or sample_width is None or frame_rate is None:
        return b""

    with tempfile.TemporaryDirectory() as temp_dir:
        input_wav_path = Path(temp_dir) / "voicevox_input.wav"
        output_opus_path = Path(temp_dir) / "voicevox_output.opus"

        with wave.open(str(input_wav_path), "wb") as wav_writer:
            wav_writer.setnchannels(channels)
            wav_writer.setsampwidth(sample_width)
            wav_writer.setframerate(frame_rate)
            wav_writer.writeframes(b"".join(pcm_frames))

        try:
            subprocess.run(
                [
                    "opusenc",
                    "--quiet",
                    str(input_wav_path),
                    str(output_opus_path),
                ],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("opusenc command was not found") from exc
        except subprocess.CalledProcessError as exc:
            stderr_output = exc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"opusenc failed: {stderr_output}") from exc

        return output_opus_path.read_bytes()


async def fetch_voicevox_speakers(voicevox_engine_url: str) -> list[dict[str, Any]]:
    speakers_url = f"{voicevox_engine_url}/speakers"
    try:
        async with ClientSession() as session:
            async with session.get(speakers_url) as response:
                if response.status >= 400:
                    error_body = await response.text()
                    raise web.HTTPBadGateway(
                        reason=f"VoiceVox speakers request failed ({response.status}): {error_body}"
                    )
                payload = await response.json()
    except ClientError as exc:
        raise web.HTTPBadGateway(reason=f"VoiceVox is unavailable: {exc}") from exc

    if not isinstance(payload, list):
        raise web.HTTPBadGateway(reason="VoiceVox speakers response was not a list")

    speaker_items: list[dict[str, Any]] = []
    for speaker in payload:
        if not isinstance(speaker, dict):
            continue
        styles = speaker.get("styles") or []
        if not isinstance(styles, list) or not styles:
            continue

        style_items: list[dict[str, Any]] = []
        for style in styles:
            if not isinstance(style, dict):
                continue
            style_id = style.get("id")
            if not isinstance(style_id, int):
                continue
            style_items.append({"id": style_id, "name": style.get("name") or "Unknown"})

        if not style_items:
            continue

        speaker_items.append(
            {
                "name": speaker.get("name") or "Unknown",
                "speaker_id": style_items[0]["id"],
                "styles": style_items,
            }
        )

    return speaker_items


class SynthesisManager:
    def __init__(self, voicevox_engine_url: str) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._stop_events: dict[str, asyncio.Event] = {}
        self._history_lock = asyncio.Lock()
        self._voicevox_engine_url = voicevox_engine_url

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

    async def start(self, text: str, voice: str, rate: str, pitch: str, engine: str) -> dict[str, Any]:
        record_id = uuid4().hex
        stop_event = asyncio.Event()
        self._stop_events[record_id] = stop_event

        record = {
            "id": record_id,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "text": text,
            "engine": engine,
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
                engine=engine,
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
        engine: str,
        stop_event: asyncio.Event,
    ) -> None:
        audio_data = b""
        try:
            if engine == "voicevox":
                try:
                    speaker_id = int(voice)
                except ValueError as exc:
                    raise RuntimeError("voice must be a valid VoiceVox speaker id") from exc
                audio_data = await synthesize_speech_voicevox(
                    text,
                    speaker_id,
                    self._voicevox_engine_url,
                    stop_event,
                )
            else:
                audio_data = await synthesize_speech_iter(text, voice, rate, pitch, stop_event)

            filename = None
            if audio_data:
                filename = f"{record_id}.opus" if engine == "voicevox" else f"{record_id}.mp3"
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
                filename = f"{record_id}.opus" if engine == "voicevox" else f"{record_id}.mp3"
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


async def index_handler(_: web.Request) -> web.FileResponse:
    return web.FileResponse(FRONTEND_DIR / "index.html")


async def health_handler(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


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

    content_type = "audio/ogg" if audio_path.suffix.lower() == ".opus" else "audio/mpeg"
    return web.FileResponse(audio_path, headers={"Content-Type": content_type})


async def api_command_handler(request: web.Request) -> web.Response:
    manager: SynthesisManager = request.app["synthesis_manager"]
    voicevox_engine_url = request.app["voicevox_engine_url"]

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        raise web.HTTPBadRequest(reason="Invalid JSON payload")

    action = (payload.get("action") or "").strip()
    if not action:
        raise web.HTTPBadRequest(reason="action is required")

    if action == "synthesize":
        text = (payload.get("text") or "").strip()
        engine = (payload.get("engine") or DEFAULT_ENGINE).strip()
        voice = payload.get("voice") or "en-US-JennyNeural"
        rate = payload.get("rate") or "+0%"
        pitch = payload.get("pitch") or "+0Hz"

        if not text:
            raise web.HTTPBadRequest(reason="Text is required")

        if engine not in {"edge-tts", "voicevox"}:
            raise web.HTTPBadRequest(reason="Unsupported engine")

        if engine == "voicevox" and not str(voice).strip():
            raise web.HTTPBadRequest(reason="voice is required for voicevox")

        record = await manager.start(text=text, voice=str(voice), rate=rate, pitch=pitch, engine=engine)
        return web.json_response({"item": record}, status=202)

    if action == "voicevox_speakers":
        return web.json_response({"items": await fetch_voicevox_speakers(voicevox_engine_url)})

    if action == "stop":
        record_id = (payload.get("record_id") or "").strip()
        if not record_id:
            raise web.HTTPBadRequest(reason="record_id is required")

        did_stop = await manager.stop(record_id)
        if not did_stop:
            raise web.HTTPNotFound(reason="No running synthesis with this record_id")
        return web.json_response({"ok": True, "record_id": record_id})

    if action == "history_list":
        return web.json_response({"items": load_history_records()})

    raise web.HTTPBadRequest(reason="Unsupported action")


def create_app(voicevox_engine_url: str, serve_frontend: bool = True) -> web.Application:
    app = web.Application()
    app["synthesis_manager"] = SynthesisManager(voicevox_engine_url=voicevox_engine_url)
    app["voicevox_engine_url"] = voicevox_engine_url

    if serve_frontend:
        app.router.add_get("/", index_handler)
        app.router.add_static("/frontend", FRONTEND_DIR)

    app.router.add_get("/health", health_handler)
    app.router.add_post("/api/command", api_command_handler)
    app.router.add_get("/api/history/{record_id}/audio", history_audio_handler)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edge TTS HTTP server")
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
    parser.add_argument(
        "--voicevox-engine-url",
        default=DEFAULT_VOICEVOX_ENGINE_URL,
        help=f"VoiceVox engine base URL (default: {DEFAULT_VOICEVOX_ENGINE_URL})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    web.run_app(
        create_app(
            voicevox_engine_url=args.voicevox_engine_url,
            serve_frontend=not args.no_frontend,
        ),
        host=args.host,
        port=args.port,
    )
