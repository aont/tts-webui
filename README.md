# VOICEVOX WebUI (aiohttp + static HTML/JS)

This project provides:

- **Python `aiohttp` proxy backend**
- **Static frontend** (HTML + JavaScript only, no Node.js)

The frontend sends text to the backend. The backend:

1. Converts Markdown/HTML input to plain text.
2. Splits long text into chunks.
3. Calls VOICEVOX `/audio_query` + `/synthesis` for each chunk.
4. Concatenates resulting WAV chunks into a single WAV.
5. Returns the merged audio for playback/download.

## API

### `POST /api/synthesize`

Request JSON:

```json
{
  "text": "# Hello\nThis is markdown.",
  "input_format": "auto",
  "speaker": 1,
  "max_chunk_chars": 120,
  "speed_scale": 1.0,
  "pitch_scale": 0.0
}
```

- `input_format`: `auto | text | markdown | html`
- `max_chunk_chars`: chunk size for long text
- `speed_scale`: speaking speed (`0.5` - `2.0`)
- `pitch_scale`: speaking pitch (`-0.15` - `0.15`)

Response:

- `audio/wav` binary
- Headers:
  - `X-Chunk-Count`
  - `X-Normalized-Length`

### `GET /api/speakers`

Returns VOICEVOX speaker/style list.

### `GET /api/health`

Simple health response.


## Frontend behavior

- You can set a custom backend endpoint URL from the frontend (empty uses same-origin `/api`).
- Frontend form values are persisted in `localStorage` and restored on reload.

## Local run

```bash
pip install -r requirements.txt
VOICEVOX_BASE_URL=http://localhost:50021 python backend/app.py --port 8080
```

(Requires VOICEVOX Engine running separately.)

### Run backend only (without serving frontend)

If you serve `frontend/` from another server, start the backend with:

```bash
VOICEVOX_BASE_URL=http://localhost:50021 python backend/app.py --no-frontend --port 8080
```

Available flags:

- `--host` (default: `HOST` env var or `0.0.0.0`)
- `--port` (default: `PORT` env var or `8080`)
- `--no-frontend` (serve API only)
