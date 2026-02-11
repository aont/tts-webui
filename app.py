import io
import os
import re
import wave
from typing import List

import requests
from flask import Flask, Response, jsonify, request, send_from_directory

VOICEVOX_BASE_URL = os.getenv("VOICEVOX_BASE_URL", "http://localhost:50021")
MAX_CHARS_PER_CHUNK = int(os.getenv("MAX_CHARS_PER_CHUNK", "180"))

app = Flask(__name__, static_folder="static")


def split_text(text: str, max_chars: int) -> List[str]:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return [stripped]

    sentences = [s.strip() for s in re.split(r"(?<=[。！？!?\.])\s+|\n+", stripped) if s.strip()]
    if not sentences:
        sentences = [stripped]

    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(sentence), max_chars):
                chunks.append(sentence[i : i + max_chars])
            continue

        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def voicevox_synthesize_chunk(text: str, speaker: int) -> bytes:
    query_res = requests.post(
        f"{VOICEVOX_BASE_URL}/audio_query",
        params={"text": text, "speaker": speaker},
        timeout=30,
    )
    query_res.raise_for_status()

    synth_res = requests.post(
        f"{VOICEVOX_BASE_URL}/synthesis",
        params={"speaker": speaker},
        json=query_res.json(),
        timeout=60,
    )
    synth_res.raise_for_status()

    return synth_res.content


def concatenate_wavs(wavs: List[bytes]) -> bytes:
    if not wavs:
        raise ValueError("No audio chunks to concatenate")

    output = io.BytesIO()
    params = None

    with wave.open(output, "wb") as out_wav:
        for raw in wavs:
            with wave.open(io.BytesIO(raw), "rb") as chunk:
                if params is None:
                    params = chunk.getparams()
                    out_wav.setparams(params)
                else:
                    if chunk.getparams()[:4] != params[:4]:
                        raise ValueError("WAV chunk format mismatch")
                out_wav.writeframes(chunk.readframes(chunk.getnframes()))

    return output.getvalue()


@app.route("/")
def root() -> Response:
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/synthesize", methods=["POST"])
def synthesize() -> Response:
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    speaker = int(payload.get("speaker", 1))

    if not text:
        return jsonify({"error": "text is required"}), 400

    chunks = split_text(text, MAX_CHARS_PER_CHUNK)

    try:
        wav_chunks = [voicevox_synthesize_chunk(chunk, speaker) for chunk in chunks]
        merged = concatenate_wavs(wav_chunks)
    except requests.RequestException as exc:
        return jsonify({"error": f"VOICEVOX request failed: {exc}"}), 502
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 500

    return Response(
        merged,
        mimetype="audio/wav",
        headers={"Content-Disposition": "inline; filename=voicevox_output.wav"},
    )


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    app.run(host=host, port=port)
