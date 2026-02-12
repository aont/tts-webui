# Edge TTS Web UI (HTTP API)

A minimal demo app using:

- **Backend**: Python, `aiohttp`, `edge-tts`
- **Frontend**: plain HTML/JS
- **API transport**: HTTP (`GET`/`POST`)

## Features

- Enter text in the browser
- Send synthesis/stop/history commands to a **single endpoint** (`POST /api/command`)
- Backend generates speech with `edge-tts`
- Long input is automatically split into manageable chunks, synthesized per chunk, and concatenated into a single MP3
- Synthesized audio is served as raw binary MP3 (`GET /api/history/{record_id}/audio`)
- Frontend can play the MP3 and download it
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
- `--no-frontend`: do not serve `/` and `/frontend/*`; only backend endpoints are available

## API

### Unified command endpoint

`POST /api/command`

#### Start synthesis

```json
{
  "action": "synthesize",
  "text": "Hello from Edge TTS",
  "voice": "en-US-JennyNeural",
  "rate": "+0%",
  "pitch": "+0Hz"
}
```

#### Stop synthesis

```json
{
  "action": "stop",
  "record_id": "<record_id>"
}
```

#### List history

```json
{
  "action": "history_list"
}
```

### Binary audio endpoint

`GET /api/history/{record_id}/audio`

- Returns `audio/mpeg` as raw binary (not base64 JSON payload).
