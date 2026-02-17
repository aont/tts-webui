import argparse
import asyncio
import math
from datetime import datetime, timezone
import io
import json
import logging
from pathlib import Path
import subprocess
import tempfile
import re
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
DEFAULT_PYAITALK_API_URL = "http://127.0.0.1:8080"
DEFAULT_LOG_LEVEL = "DEBUG"


logger = logging.getLogger(__name__)

SENTENCE_END_CHARS = {"。", "．", ".", "!", "?", "！", "？", "\n"}
TRAILING_CLOSERS = re.compile(r"[\s\"'\)\]\}」』】》〉》）】”’]")


class SynthesisStoppedError(Exception):
    pass


@web.middleware
async def cors_middleware(request: web.Request, handler):
    allow_origin = request.app["allow_origin"]
    allow_headers = "Content-Type"
    allow_methods = "GET,POST,OPTIONS"

    if request.method == "OPTIONS":
        response = web.Response(status=204)
    else:
        response = await handler(request)

    response.headers["Access-Control-Allow-Origin"] = allow_origin
    response.headers["Access-Control-Allow-Methods"] = allow_methods
    response.headers["Access-Control-Allow-Headers"] = allow_headers
    response.headers["Access-Control-Max-Age"] = "600"
    return response


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
        logger.exception("Failed to load history records from %s", HISTORY_FILE)
        return []

    if not isinstance(records, list):
        return []

    return [record for record in records if isinstance(record, dict)]


def save_history_records(records: list[dict[str, Any]]) -> None:
    ensure_history_storage()
    logger.debug("Persisting %d history records to %s", len(records), HISTORY_FILE)
    HISTORY_FILE.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def trim_history(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stale_records = records[MAX_HISTORY_ITEMS:]
    if stale_records:
        logger.debug("Trimming %d stale history record(s)", len(stale_records))
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
    """Split text using sentence-end candidates near evenly spaced targets."""
    stripped_text = text.strip()
    if not stripped_text:
        return []

    candidates: list[int] = []
    for i, char in enumerate(stripped_text):
        if char not in SENTENCE_END_CHARS:
            continue

        end = i + 1
        while end < len(stripped_text) and TRAILING_CLOSERS.fullmatch(stripped_text[end]):
            end += 1
        if end > 0 and end not in candidates:
            candidates.append(end)

    if len(stripped_text) not in candidates:
        candidates.append(len(stripped_text))
    candidates.sort()

    segment_count = max(1, math.ceil(len(stripped_text) / max_chars))
    target_segment_length = len(stripped_text) / segment_count

    boundaries: list[int] = []
    previous = 0

    for split_index in range(1, segment_count):
        target = target_segment_length * split_index
        max_end = min(previous + max_chars, len(stripped_text))

        selected = next(
            (
                point
                for point in candidates
                if previous < point <= max_end and point > target
            ),
            None,
        )

        if selected is None:
            selected = next(
                (point for point in candidates if previous < point <= max_end),
                max_end,
            )

        boundaries.append(selected)
        previous = selected

    boundaries.append(len(stripped_text))

    segments: list[str] = []
    start = 0
    for end in boundaries:
        segment = stripped_text[start:end].strip()
        if segment:
            segments.append(segment)
        start = end

    logger.debug(
        "Split input text into %d segment(s) (max_chars=%d, input_len=%d, target_len=%.2f)",
        len(segments),
        max_chars,
        len(stripped_text),
        target_segment_length,
    )
    return segments


def normalize_segments_payload(raw_segments: Any, fallback_text: str) -> list[str]:
    if not isinstance(raw_segments, list):
        return split_text_segments(fallback_text)

    segments: list[str] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, str):
            continue
        segment = raw_segment.strip()
        if not segment:
            continue
        if len(segment) > MAX_SEGMENT_CHARS:
            raise web.HTTPBadRequest(
                reason=f"Each segment must be <= {MAX_SEGMENT_CHARS} characters"
            )
        segments.append(segment)

    if not segments:
        return split_text_segments(fallback_text)

    logger.debug(
        "Accepted %d segment(s) from request payload (chars=%d)",
        len(segments),
        sum(len(segment) for segment in segments),
    )
    return segments


async def synthesize_speech_iter(
    segments: list[str],
    voice: str,
    rate: str,
    pitch: str,
    stop_event: asyncio.Event,
) -> bytes:
    audio_chunks: list[bytes] = []
    logger.debug(
        "Starting edge-tts synthesis (voice=%s, rate=%s, pitch=%s, text_len=%d)",
        voice,
        rate,
        pitch,
        sum(len(segment) for segment in segments),
    )

    for index, text_segment in enumerate(segments, start=1):
        if stop_event.is_set():
            raise SynthesisStoppedError("Synthesis stopped by user")

        logger.debug("edge-tts segment %d queued (segment_len=%d)", index, len(text_segment))

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
    segments: list[str],
    speaker: int,
    voicevox_engine_url: str,
    stop_event: asyncio.Event,
) -> bytes:
    if stop_event.is_set():
        raise SynthesisStoppedError("Synthesis stopped by user")

    wav_chunks: list[bytes] = []
    logger.debug(
        "Starting VoiceVox synthesis (speaker=%s, text_len=%d, engine_url=%s)",
        speaker,
        sum(len(segment) for segment in segments),
        voicevox_engine_url,
    )

    async with ClientSession() as session:
        for index, text_segment in enumerate(segments, start=1):
            if stop_event.is_set():
                raise SynthesisStoppedError("Synthesis stopped by user")

            query_url = f"{voicevox_engine_url}/audio_query"
            synth_url = f"{voicevox_engine_url}/synthesis"

            async with session.post(
                query_url,
                params={"text": text_segment, "speaker": speaker},
            ) as query_response:
                logger.debug(
                    "VoiceVox audio_query request sent (segment=%d, chars=%d, status=%d)",
                    index,
                    len(text_segment),
                    query_response.status,
                )
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
                logger.debug(
                    "VoiceVox synthesis request sent (segment=%d, status=%d)",
                    index,
                    synth_response.status,
                )
                if synth_response.status >= 400:
                    error_body = await synth_response.text()
                    raise RuntimeError(
                        f"VoiceVox synthesis failed ({synth_response.status}): {error_body}"
                    )
                wav_chunks.append(await synth_response.read())

    return encode_voicevox_wav_chunks_to_opus(wav_chunks)


async def synthesize_speech_pyaitalk(
    segments: list[str],
    character: str,
    pyaitalk_api_url: str,
    stop_event: asyncio.Event,
) -> bytes:
    if stop_event.is_set():
        raise SynthesisStoppedError("Synthesis stopped by user")

    wav_chunks: list[bytes] = []
    logger.debug(
        "Starting pyaitalk synthesis (character=%s, text_len=%d, api_url=%s)",
        character,
        sum(len(segment) for segment in segments),
        pyaitalk_api_url,
    )

    pyaitalk_api_url = pyaitalk_api_url.rstrip("/")
    synth_url = f"{pyaitalk_api_url}/synthesize"
    async with ClientSession() as session:
        for index, text_segment in enumerate(segments, start=1):
            if stop_event.is_set():
                raise SynthesisStoppedError("Synthesis stopped by user")

            payload: dict[str, Any] = {"text": text_segment, "output": "wav"}
            if character.strip():
                payload["character"] = character

            async with session.post(synth_url, json=payload) as synth_response:
                logger.debug(
                    "pyaitalk synthesis request sent (segment=%d, status=%d)",
                    index,
                    synth_response.status,
                )
                if synth_response.status >= 400:
                    error_body = await synth_response.text()
                    raise RuntimeError(
                        f"pyaitalk synthesize failed ({synth_response.status}): {error_body}"
                    )
                wav_chunks.append(await synth_response.read())

    return encode_wav_chunks_to_opus(wav_chunks, engine_name="pyaitalk")


def encode_wav_chunks_to_opus(wav_chunks: list[bytes], engine_name: str) -> bytes:
    if not wav_chunks:
        logger.debug("No %s WAV chunks were returned", engine_name)
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
                raise RuntimeError(f"{engine_name} returned WAV chunks with mismatched audio format")

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


def encode_voicevox_wav_chunks_to_opus(wav_chunks: list[bytes]) -> bytes:
    return encode_wav_chunks_to_opus(wav_chunks, engine_name="VoiceVox")


async def fetch_voicevox_speakers(voicevox_engine_url: str) -> list[dict[str, Any]]:
    speakers_url = f"{voicevox_engine_url}/speakers"
    logger.debug("Fetching VoiceVox speakers from %s", speakers_url)
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

    logger.debug("Fetched %d VoiceVox speaker entries", len(speaker_items))
    return speaker_items


async def fetch_pyaitalk_voices(pyaitalk_api_url: str) -> list[dict[str, str]]:
    pyaitalk_api_url = pyaitalk_api_url.rstrip("/")
    voices_url = f"{pyaitalk_api_url}/voice/list"
    logger.debug("Fetching pyaitalk voices from %s", voices_url)
    try:
        async with ClientSession() as session:
            async with session.get(voices_url) as response:
                if response.status >= 400:
                    error_body = await response.text()
                    raise web.HTTPBadGateway(
                        reason=f"pyaitalk voice list request failed ({response.status}): {error_body}"
                    )
                payload = await response.json()
    except ClientError as exc:
        raise web.HTTPBadGateway(reason=f"pyaitalk is unavailable: {exc}") from exc

    voice_names = payload.get("voices") if isinstance(payload, dict) else None
    if not isinstance(voice_names, list):
        raise web.HTTPBadGateway(reason="pyaitalk voice list response did not include voices")

    voices: list[dict[str, str]] = []
    for voice_name in voice_names:
        if not isinstance(voice_name, str):
            continue
        normalized_name = voice_name.strip()
        if not normalized_name:
            continue
        voices.append({"id": normalized_name, "name": normalized_name})

    logger.debug("Fetched %d pyaitalk voice entries", len(voices))
    return voices


class SynthesisManager:
    def __init__(self, voicevox_engine_url: str, pyaitalk_api_url: str) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._stop_events: dict[str, asyncio.Event] = {}
        self._history_lock = asyncio.Lock()
        self._voicevox_engine_url = voicevox_engine_url
        self._pyaitalk_api_url = pyaitalk_api_url

    async def _update_history(self, record_id: str, updates: dict[str, Any]) -> None:
        logger.debug("Updating history record %s with keys=%s", record_id, sorted(updates.keys()))
        async with self._history_lock:
            records = load_history_records()
            for record in records:
                if record.get("id") == record_id:
                    record.update(updates)
                    break
            save_history_records(trim_history(records))

    async def _create_history_record(self, payload: dict[str, Any]) -> dict[str, Any]:
        logger.debug("Creating history record %s", payload.get("id"))
        async with self._history_lock:
            records = load_history_records()
            records.insert(0, payload)
            save_history_records(trim_history(records))
        return payload

    async def start(
        self,
        text: str,
        segments: list[str],
        voice: str,
        rate: str,
        pitch: str,
        engine: str,
    ) -> dict[str, Any]:
        record_id = uuid4().hex
        stop_event = asyncio.Event()
        self._stop_events[record_id] = stop_event
        logger.debug(
            "Registering synthesis task %s (engine=%s, voice=%s, text_len=%d)",
            record_id,
            engine,
            voice,
            len(text),
        )

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
                segments=segments,
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
        segments: list[str],
        voice: str,
        rate: str,
        pitch: str,
        engine: str,
        stop_event: asyncio.Event,
    ) -> None:
        audio_data = b""
        try:
            logger.debug("Synthesis task %s started (engine=%s)", record_id, engine)
            if engine == "voicevox":
                try:
                    speaker_id = int(voice)
                except ValueError as exc:
                    raise RuntimeError("voice must be a valid VoiceVox speaker id") from exc
                audio_data = await synthesize_speech_voicevox(
                    segments,
                    speaker_id,
                    self._voicevox_engine_url,
                    stop_event,
                )
            elif engine == "pyaitalk":
                audio_data = await synthesize_speech_pyaitalk(
                    segments,
                    voice,
                    self._pyaitalk_api_url,
                    stop_event,
                )
            else:
                audio_data = await synthesize_speech_iter(segments, voice, rate, pitch, stop_event)

            filename = None
            if audio_data:
                filename = f"{record_id}.opus" if engine in {"voicevox", "pyaitalk"} else f"{record_id}.mp3"
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
            logger.debug("Synthesis task %s completed (audio_bytes=%d)", record_id, len(audio_data))
        except SynthesisStoppedError:
            filename = None
            if audio_data:
                filename = f"{record_id}.opus" if engine in {"voicevox", "pyaitalk"} else f"{record_id}.mp3"
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
            logger.debug("Synthesis task %s stopped by user", record_id)
        except Exception as exc:
            logger.exception("Synthesis task %s failed: %s", record_id, exc)
            await self._update_history(
                record_id,
                {
                    "status": "failed",
                    "error": str(exc),
                    "updated_at": now_iso(),
                },
            )
        finally:
            logger.debug("Cleaning up synthesis task %s", record_id)
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


async def parse_json_payload(request: web.Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        raise web.HTTPBadRequest(reason="Invalid JSON payload")

    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(reason="JSON object payload is required")
    return payload


async def synthesize_handler(request: web.Request) -> web.Response:
    manager: SynthesisManager = request.app["synthesis_manager"]
    payload = await parse_json_payload(request)
    logger.debug("Received /synthesize payload_keys=%s", sorted(payload.keys()))

    text = (payload.get("text") or "").strip()
    segments = normalize_segments_payload(payload.get("segments"), text)
    engine = (payload.get("engine") or DEFAULT_ENGINE).strip()
    voice = payload.get("voice") or "en-US-JennyNeural"
    rate = payload.get("rate") or "+0%"
    pitch = payload.get("pitch") or "+0Hz"

    if not text:
        raise web.HTTPBadRequest(reason="Text is required")

    if engine not in {"edge-tts", "voicevox", "pyaitalk"}:
        raise web.HTTPBadRequest(reason="Unsupported engine")

    if engine == "voicevox" and not str(voice).strip():
        raise web.HTTPBadRequest(reason="voice is required for voicevox")

    if engine == "pyaitalk" and not str(voice).strip():
        raise web.HTTPBadRequest(reason="voice is required for pyaitalk")

    record = await manager.start(
        text=text,
        segments=segments,
        voice=str(voice),
        rate=rate,
        pitch=pitch,
        engine=engine,
    )
    return web.json_response({"item": record}, status=202)


async def voicevox_speakers_handler(request: web.Request) -> web.Response:
    voicevox_engine_url = request.app["voicevox_engine_url"]
    return web.json_response({"items": await fetch_voicevox_speakers(voicevox_engine_url)})


async def pyaitalk_voices_handler(request: web.Request) -> web.Response:
    pyaitalk_api_url = request.app["pyaitalk_api_url"]
    return web.json_response({"items": await fetch_pyaitalk_voices(pyaitalk_api_url)})


async def stop_handler(request: web.Request) -> web.Response:
    manager: SynthesisManager = request.app["synthesis_manager"]
    payload = await parse_json_payload(request)
    record_id = (payload.get("record_id") or "").strip()
    if not record_id:
        raise web.HTTPBadRequest(reason="record_id is required")

    did_stop = await manager.stop(record_id)
    if not did_stop:
        raise web.HTTPNotFound(reason="No running synthesis with this record_id")
    return web.json_response({"ok": True, "record_id": record_id})


async def history_list_handler(_: web.Request) -> web.Response:
    return web.json_response({"items": load_history_records()})


def create_app(
    voicevox_engine_url: str,
    pyaitalk_api_url: str,
    allow_origin: str,
    serve_frontend: bool = True,
) -> web.Application:
    logger.debug(
        "Creating app (serve_frontend=%s, voicevox_engine_url=%s, pyaitalk_api_url=%s, allow_origin=%s)",
        serve_frontend,
        voicevox_engine_url,
        pyaitalk_api_url,
        allow_origin,
    )
    app = web.Application(middlewares=[cors_middleware])
    app["synthesis_manager"] = SynthesisManager(
        voicevox_engine_url=voicevox_engine_url,
        pyaitalk_api_url=pyaitalk_api_url,
    )
    app["voicevox_engine_url"] = voicevox_engine_url
    app["pyaitalk_api_url"] = pyaitalk_api_url
    app["allow_origin"] = allow_origin

    if serve_frontend:
        app.router.add_get("/", index_handler)
        app.router.add_static("/frontend", FRONTEND_DIR)

    app.router.add_get("/health", health_handler)
    app.router.add_post("/synthesize", synthesize_handler)
    app.router.add_route("OPTIONS", "/synthesize", synthesize_handler)
    app.router.add_get("/voicevox/speakers", voicevox_speakers_handler)
    app.router.add_route("OPTIONS", "/voicevox/speakers", voicevox_speakers_handler)
    app.router.add_get("/pyaitalk/voices", pyaitalk_voices_handler)
    app.router.add_route("OPTIONS", "/pyaitalk/voices", pyaitalk_voices_handler)
    app.router.add_post("/stop", stop_handler)
    app.router.add_route("OPTIONS", "/stop", stop_handler)
    app.router.add_get("/history", history_list_handler)
    app.router.add_route("OPTIONS", "/history", history_list_handler)
    app.router.add_get("/history/{record_id}/audio", history_audio_handler)
    app.router.add_route("OPTIONS", "/history/{record_id}/audio", history_audio_handler)
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
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        help=f"Logging level (default: {DEFAULT_LOG_LEVEL})",
    )
    parser.add_argument(
        "--voicevox-engine-url",
        default=DEFAULT_VOICEVOX_ENGINE_URL,
        help=f"VoiceVox engine base URL (default: {DEFAULT_VOICEVOX_ENGINE_URL})",
    )
    parser.add_argument(
        "--pyaitalk-api-url",
        default=DEFAULT_PYAITALK_API_URL,
        help=f"pyaitalk API base URL (default: {DEFAULT_PYAITALK_API_URL})",
    )
    parser.add_argument(
        "--allow-origin",
        default="*",
        help="CORS Access-Control-Allow-Origin value (default: *)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.DEBUG),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.debug("Starting server with args=%s", vars(args))
    web.run_app(
        create_app(
            voicevox_engine_url=args.voicevox_engine_url,
            pyaitalk_api_url=args.pyaitalk_api_url,
            allow_origin=args.allow_origin,
            serve_frontend=not args.no_frontend,
        ),
        host=args.host,
        port=args.port,
    )
