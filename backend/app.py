import argparse
import asyncio
import datetime as dt
import io
import os
import re
import subprocess
import tempfile
import uuid
import wave
from collections import deque
from typing import Iterable, List

import markdown as md_lib
from aiohttp import ClientError, ClientSession, ClientTimeout, web
from bs4 import BeautifulSoup

VOICEVOX_BASE_URL = "http://localhost:50021"
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "120"))
MAX_HISTORY_ITEMS = int(os.getenv("MAX_HISTORY_ITEMS", "20"))


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def history_to_json(item: dict) -> dict:
    return {
        "id": item["id"],
        "created_at": item["created_at"],
        "updated_at": item.get("updated_at", item["created_at"]),
        "status": item.get("status", "completed"),
        "error": item.get("error"),
        "text": item["text"],
        "speaker": item["speaker"],
        "input_format": item["input_format"],
        "max_chunk_chars": item["max_chunk_chars"],
        "speed_scale": item["speed_scale"],
        "pitch_scale": item["pitch_scale"],
        "chunk_count": item["chunk_count"],
        "normalized_length": item["normalized_length"],
    }


def html_to_text(html: str) -> str:
    doc = BeautifulSoup(html, "html.parser")
    text = doc.get_text(separator="\n")
    return re.sub(r"\s+\n", "\n", text).strip()


def markdown_to_text(markdown_text: str) -> str:
    html = md_lib.markdown(markdown_text)
    return html_to_text(html)


def normalize_text(raw_text: str, input_format: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return ""

    fmt = (input_format or "auto").lower()

    if fmt == "markdown":
        return markdown_to_text(text)
    if fmt == "html":
        return html_to_text(text)
    if fmt == "text":
        return text

    if re.search(r"<[^>]+>", text):
        return html_to_text(text)

    markdown_markers = [r"(^|\n)#{1,6}\s", r"\*\*.*?\*\*", r"`[^`]+`", r"(^|\n)[\-\*]\s"]
    if any(re.search(marker, text, flags=re.MULTILINE | re.DOTALL) for marker in markdown_markers):
        return markdown_to_text(text)

    return text


def split_text(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    sentence_like = re.split(r"(?<=[。！？.!?\n])", text)
    chunks: List[str] = []
    current = ""

    for part in sentence_like:
        if not part:
            continue

        if len(part) > max_chars:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            chunks.extend(
                part[i : i + max_chars].strip()
                for i in range(0, len(part), max_chars)
                if part[i : i + max_chars].strip()
            )
            continue

        if len(current) + len(part) > max_chars:
            if current.strip():
                chunks.append(current.strip())
            current = part
        else:
            current += part

    if current.strip():
        chunks.append(current.strip())

    return chunks


async def voicevox_query(
    session: ClientSession, text: str, speaker: int, speed_scale: float, pitch_scale: float
) -> dict:
    async with session.post(
        f"{VOICEVOX_BASE_URL}/audio_query",
        params={"text": text, "speaker": speaker},
    ) as resp:
        resp.raise_for_status()
        query = await resp.json()

    query["speedScale"] = speed_scale
    query["pitchScale"] = pitch_scale
    return query


async def voicevox_synthesis(session: ClientSession, query: dict, speaker: int) -> bytes:
    async with session.post(
        f"{VOICEVOX_BASE_URL}/synthesis",
        params={"speaker": speaker},
        json=query,
    ) as resp:
        resp.raise_for_status()
        return await resp.read()


def concat_wavs(wav_blobs: Iterable[bytes]) -> bytes:
    merged = io.BytesIO()
    params = None
    frames = []

    for blob in wav_blobs:
        with wave.open(io.BytesIO(blob), "rb") as reader:
            if params is None:
                params = reader.getparams()
            elif reader.getparams()[:4] != params[:4]:
                raise ValueError("WAV format mismatch while concatenating synthesized chunks")
            frames.append(reader.readframes(reader.getnframes()))

    if params is None:
        raise ValueError("No WAV audio generated")

    with wave.open(merged, "wb") as writer:
        writer.setparams(params)
        for frame in frames:
            writer.writeframes(frame)

    return merged.getvalue()


def encode_wav_to_opus(wav_blob: bytes) -> bytes:
    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = os.path.join(temp_dir, "input.wav")
        opus_path = os.path.join(temp_dir, "output.opus")

        with open(wav_path, "wb") as wav_file:
            wav_file.write(wav_blob)

        try:
            subprocess.run(
                [
                    "opusenc",
                    "--quiet",
                    wav_path,
                    opus_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("opusenc command is not available") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(f"failed to encode audio to opus: {stderr}") from exc

        with open(opus_path, "rb") as opus_file:
            return opus_file.read()


async def handle_index(_: web.Request) -> web.Response:
    return web.FileResponse("frontend/index.html")


async def handle_health(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "voicevox_base_url": VOICEVOX_BASE_URL})



def add_history_item(app: web.Application, item: dict) -> None:
    history_items = app["history_items"]
    history_audio = app["history_audio"]

    history_items.appendleft(item)
    history_audio[item["id"]] = item["audio"]

    while len(history_items) > app["max_history_items"]:
        evicted = history_items.pop()
        history_audio.pop(evicted["id"], None)
        running_task = app["synthesis_tasks"].pop(evicted["id"], None)
        if running_task and not running_task.done():
            running_task.cancel()


def update_history_item(app: web.Application, item_id: str, **fields: object) -> dict | None:
    history_items = app["history_items"]
    for item in history_items:
        if item["id"] == item_id:
            item.update(fields)
            item["updated_at"] = now_iso()
            return item
    return None


async def handle_history(request: web.Request) -> web.Response:
    history_items = request.app["history_items"]
    return web.json_response({"items": [history_to_json(item) for item in history_items]})


async def handle_history_audio(request: web.Request) -> web.Response:
    item_id = request.match_info["item_id"]
    item = next((history_item for history_item in request.app["history_items"] if history_item["id"] == item_id), None)
    if item is None:
        return web.json_response({"error": "history item not found"}, status=404)

    if item.get("status") != "completed":
        return web.json_response({"error": f"audio is not available for status '{item.get('status')}'"}, status=409)

    audio = request.app["history_audio"].get(item_id)
    if audio is None:
        return web.json_response({"error": "history item not found"}, status=404)

    headers = {"Content-Disposition": f"attachment; filename=voicevox-history-{item_id}.opus"}
    return web.Response(body=audio, content_type="audio/ogg", headers=headers)


async def run_synthesis_job(app: web.Application, job_id: str, chunks: List[str], speaker: int, speed_scale: float, pitch_scale: float) -> None:
    timeout = ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    try:
        async with ClientSession(timeout=timeout) as session:
            queries = [
                voicevox_query(session, chunk, speaker, speed_scale, pitch_scale)
                for chunk in chunks
            ]
            query_results = await asyncio.gather(*queries)

            synthesis_tasks = [
                voicevox_synthesis(session, query_result, speaker)
                for query_result in query_results
            ]
            wav_parts = await asyncio.gather(*synthesis_tasks)

        merged_wav = concat_wavs(wav_parts)
        merged_opus = encode_wav_to_opus(merged_wav)
        app["history_audio"][job_id] = merged_opus
        update_history_item(app, job_id, status="completed")
    except asyncio.CancelledError:
        update_history_item(app, job_id, status="stopped", error="stopped by user")
        raise
    except ClientError as exc:
        update_history_item(
            app,
            job_id,
            status="failed",
            error=f"failed to connect to VOICEVOX engine at {VOICEVOX_BASE_URL}: {exc}",
        )
    except RuntimeError as exc:
        update_history_item(app, job_id, status="failed", error=str(exc))
    except Exception as exc:
        update_history_item(app, job_id, status="failed", error=f"unexpected error: {exc}")
    finally:
        app["synthesis_tasks"].pop(job_id, None)


async def handle_synthesize_start(request: web.Request) -> web.Response:
    payload = await request.json()
    input_format_value = payload.get("input_format", "auto")
    text = normalize_text(payload.get("text", ""), input_format_value)
    speaker = int(payload.get("speaker", 1))
    max_chunk_chars = int(payload.get("max_chunk_chars", MAX_CHUNK_CHARS))
    speed_scale = float(payload.get("speed_scale", 1.0))
    pitch_scale = float(payload.get("pitch_scale", 0.0))

    if not text:
        return web.json_response({"error": "text is empty after preprocessing"}, status=400)
    if max_chunk_chars < 8:
        return web.json_response({"error": "max_chunk_chars must be >= 8"}, status=400)
    if not (0.5 <= speed_scale <= 2.0):
        return web.json_response({"error": "speed_scale must be between 0.5 and 2.0"}, status=400)
    if not (-0.15 <= pitch_scale <= 0.15):
        return web.json_response({"error": "pitch_scale must be between -0.15 and 0.15"}, status=400)

    chunks = split_text(text, max_chunk_chars)
    history_id = str(uuid.uuid4())
    add_history_item(
        request.app,
        {
            "id": history_id,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "status": "in_progress",
            "error": None,
            "text": text,
            "speaker": speaker,
            "input_format": input_format_value,
            "max_chunk_chars": max_chunk_chars,
            "speed_scale": speed_scale,
            "pitch_scale": pitch_scale,
            "chunk_count": len(chunks),
            "normalized_length": len(text),
            "audio": b"",
        },
    )

    task = asyncio.create_task(
        run_synthesis_job(request.app, history_id, chunks, speaker, speed_scale, pitch_scale)
    )
    request.app["synthesis_tasks"][history_id] = task

    return web.json_response({"id": history_id, "status": "in_progress"}, status=202)


async def handle_stop_synthesis(request: web.Request) -> web.Response:
    item_id = request.match_info["item_id"]
    item = update_history_item(request.app, item_id)
    if item is None:
        return web.json_response({"error": "history item not found"}, status=404)

    if item.get("status") != "in_progress":
        return web.json_response({"error": f"cannot stop synthesis with status '{item.get('status')}'"}, status=409)

    task = request.app["synthesis_tasks"].get(item_id)
    if task is None:
        update_history_item(request.app, item_id, status="stopped", error="stopped by user")
        return web.json_response({"id": item_id, "status": "stopped"})

    task.cancel()
    return web.json_response({"id": item_id, "status": "stopping"})


async def handle_synthesize(request: web.Request) -> web.Response:
    payload = await request.json()
    input_format_value = payload.get("input_format", "auto")
    text = normalize_text(payload.get("text", ""), input_format_value)
    speaker = int(payload.get("speaker", 1))
    max_chunk_chars = int(payload.get("max_chunk_chars", MAX_CHUNK_CHARS))
    speed_scale = float(payload.get("speed_scale", 1.0))
    pitch_scale = float(payload.get("pitch_scale", 0.0))

    if not text:
        return web.json_response({"error": "text is empty after preprocessing"}, status=400)
    if max_chunk_chars < 8:
        return web.json_response({"error": "max_chunk_chars must be >= 8"}, status=400)
    if not (0.5 <= speed_scale <= 2.0):
        return web.json_response({"error": "speed_scale must be between 0.5 and 2.0"}, status=400)
    if not (-0.15 <= pitch_scale <= 0.15):
        return web.json_response({"error": "pitch_scale must be between -0.15 and 0.15"}, status=400)

    chunks = split_text(text, max_chunk_chars)

    timeout = ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    try:
        async with ClientSession(timeout=timeout) as session:
            queries = [
                voicevox_query(session, chunk, speaker, speed_scale, pitch_scale)
                for chunk in chunks
            ]
            query_results = await asyncio.gather(*queries)

            synthesis_tasks = [
                voicevox_synthesis(session, query_result, speaker)
                for query_result in query_results
            ]
            wav_parts = await asyncio.gather(*synthesis_tasks)

        merged_wav = concat_wavs(wav_parts)
        merged_opus = encode_wav_to_opus(merged_wav)
    except ClientError as exc:
        return web.json_response(
            {"error": f"failed to connect to VOICEVOX engine at {VOICEVOX_BASE_URL}: {exc}"},
            status=502,
        )
    except RuntimeError as exc:
        return web.json_response({"error": str(exc)}, status=500)

    history_id = str(uuid.uuid4())
    add_history_item(
        request.app,
        {
            "id": history_id,
            "created_at": now_iso(),
            "text": text,
            "speaker": speaker,
            "input_format": input_format_value,
            "max_chunk_chars": max_chunk_chars,
            "speed_scale": speed_scale,
            "pitch_scale": pitch_scale,
            "chunk_count": len(chunks),
            "normalized_length": len(text),
            "audio": merged_opus,
        },
    )

    headers = {
        "Content-Disposition": "attachment; filename=voicevox-output.opus",
        "X-Chunk-Count": str(len(chunks)),
        "X-Normalized-Length": str(len(text)),
        "X-History-Id": history_id,
    }
    return web.Response(body=merged_opus, content_type="audio/ogg", headers=headers)


async def handle_speakers(_: web.Request) -> web.Response:
    timeout = ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    try:
        async with ClientSession(timeout=timeout) as session:
            async with session.get(f"{VOICEVOX_BASE_URL}/speakers") as resp:
                resp.raise_for_status()
                data = await resp.json()
        return web.json_response(data)
    except ClientError as exc:
        return web.json_response(
            {"error": f"failed to connect to VOICEVOX engine at {VOICEVOX_BASE_URL}: {exc}"},
            status=502,
        )


def build_app(serve_frontend: bool = True) -> web.Application:
    app = web.Application()
    app.router.add_get("/api/health", handle_health)
    app["history_items"] = deque()
    app["history_audio"] = {}
    app["max_history_items"] = MAX_HISTORY_ITEMS
    app["synthesis_tasks"] = {}

    app.router.add_get("/api/speakers", handle_speakers)
    app.router.add_get("/api/history", handle_history)
    app.router.add_get("/api/history/{item_id}/audio", handle_history_audio)
    app.router.add_post("/api/synthesize", handle_synthesize)
    app.router.add_post("/api/synthesize/start", handle_synthesize_start)
    app.router.add_post("/api/synthesize/{item_id}/stop", handle_stop_synthesis)

    if serve_frontend:
        app.router.add_get("/", handle_index)
        app.router.add_static("/", "frontend", show_index=True)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VOICEVOX WebUI backend server")
    parser.add_argument(
        "--voicevox-base-url",
        default="http://localhost:50021",
        help="Base URL of the VOICEVOX engine (default: %(default)s)",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8080")),
        help="Port to listen on (default: %(default)s)",
    )
    parser.add_argument(
        "--no-frontend",
        action="store_true",
        help="Disable serving static frontend files from this backend",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    VOICEVOX_BASE_URL = args.voicevox_base_url.rstrip("/")
    app = build_app(serve_frontend=not args.no_frontend)
    web.run_app(app, host=args.host, port=args.port)
