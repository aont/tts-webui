import argparse
import asyncio
import io
import os
import re
import wave
from typing import Iterable, List

import markdown as md_lib
from aiohttp import ClientError, ClientSession, ClientTimeout, web
from bs4 import BeautifulSoup

VOICEVOX_BASE_URL = os.getenv("VOICEVOX_BASE_URL", "http://localhost:50021").rstrip("/")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "120"))


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


async def voicevox_query(session: ClientSession, text: str, speaker: int) -> dict:
    async with session.post(
        f"{VOICEVOX_BASE_URL}/audio_query",
        params={"text": text, "speaker": speaker},
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


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


async def handle_index(_: web.Request) -> web.Response:
    return web.FileResponse("frontend/index.html")


async def handle_health(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "voicevox_base_url": VOICEVOX_BASE_URL})


async def handle_synthesize(request: web.Request) -> web.Response:
    payload = await request.json()
    text = normalize_text(payload.get("text", ""), payload.get("input_format", "auto"))
    speaker = int(payload.get("speaker", 1))
    max_chunk_chars = int(payload.get("max_chunk_chars", MAX_CHUNK_CHARS))

    if not text:
        return web.json_response({"error": "text is empty after preprocessing"}, status=400)
    if max_chunk_chars < 8:
        return web.json_response({"error": "max_chunk_chars must be >= 8"}, status=400)

    chunks = split_text(text, max_chunk_chars)

    timeout = ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    try:
        async with ClientSession(timeout=timeout) as session:
            queries = [voicevox_query(session, chunk, speaker) for chunk in chunks]
            query_results = await asyncio.gather(*queries)

            synthesis_tasks = [
                voicevox_synthesis(session, query_result, speaker)
                for query_result in query_results
            ]
            wav_parts = await asyncio.gather(*synthesis_tasks)

        merged_wav = concat_wavs(wav_parts)
    except ClientError as exc:
        return web.json_response(
            {"error": f"failed to connect to VOICEVOX engine at {VOICEVOX_BASE_URL}: {exc}"},
            status=502,
        )

    headers = {
        "Content-Disposition": "attachment; filename=voicevox-output.wav",
        "X-Chunk-Count": str(len(chunks)),
        "X-Normalized-Length": str(len(text)),
    }
    return web.Response(body=merged_wav, content_type="audio/wav", headers=headers)


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
    app.router.add_get("/api/speakers", handle_speakers)
    app.router.add_post("/api/synthesize", handle_synthesize)

    if serve_frontend:
        app.router.add_get("/", handle_index)
        app.router.add_static("/", "frontend", show_index=True)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VOICEVOX WebUI backend server")
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
    app = build_app(serve_frontend=not args.no_frontend)
    web.run_app(app, host=args.host, port=args.port)
