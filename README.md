# Edge TTS Web UI (WebSocket)

A minimal demo app using:

- **Backend**: Python, `aiohttp`, `edge-tts`
- **Frontend**: plain HTML/JS
- **Realtime channel**: **WebSocket only** via `/ws`

## Features

- Enter text in the browser
- Send synthesis requests to backend over WebSocket
- Backend generates speech with `edge-tts`
- Long input is automatically split into manageable chunks, synthesized per chunk, and concatenated into a single MP3
- Frontend can play the returned audio and download it as an MP3
- Backend host/port and frontend serving can be controlled via command-line flags

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py --port 8080
```

Open: <http://localhost:8080>

## CLI options

```bash
python app.py --host 0.0.0.0 --port 9000
```

- `--host`: host/interface to bind (default `0.0.0.0`)
- `--port`: port to bind (default `8080`)
- `--no-frontend`: do not serve `/` and `/frontend/*`; only backend endpoints (`/ws`, `/health`) are available

If you serve the frontend from another server, set **Backend WS URL** in the UI to point to this backend (for example `ws://localhost:9000/ws`).

## WebSocket message format

### Request

```json
{
  "action": "synthesize",
  "text": "Hello from Edge TTS",
  "voice": "en-US-JennyNeural"
}
```

### Audio response

```json
{
  "type": "audio",
  "format": "audio/mpeg",
  "filename": "speech.mp3",
  "audio_base64": "..."
}
```
