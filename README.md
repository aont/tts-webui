# VOICEVOX WebUI (aiohttp + static HTML/JS)

This project provides:

- **VOICEVOX Engine in Docker** (`voicevox/voicevox_engine`)
- **Python `aiohttp` proxy backend**
- **Static frontend** (HTML + JavaScript only, no Node.js)

The frontend sends text to the backend. The backend:

1. Converts Markdown/HTML input to plain text.
2. Splits long text into chunks.
3. Calls VOICEVOX `/audio_query` + `/synthesis` for each chunk.
4. Concatenates resulting WAV chunks into a single WAV.
5. Returns the merged audio for playback/download.

## Run with Docker Compose

```bash
docker compose up --build
```

Then open:

- `http://localhost:8080`

## API

### `POST /api/synthesize`

Request JSON:

```json
{
  "text": "# Hello\nThis is markdown.",
  "input_format": "auto",
  "speaker": 1,
  "max_chunk_chars": 120
}
```

- `input_format`: `auto | text | markdown | html`
- `max_chunk_chars`: chunk size for long text

Response:

- `audio/wav` binary
- Headers:
  - `X-Chunk-Count`
  - `X-Normalized-Length`

### `GET /api/speakers`

Returns VOICEVOX speaker/style list.

### `GET /api/health`

Simple health response.

## Local run without Docker Compose

```bash
pip install -r requirements.txt
VOICEVOX_BASE_URL=http://localhost:50021 python backend/app.py
```

(Requires VOICEVOX Engine running separately.)
