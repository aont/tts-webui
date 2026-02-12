# TTS Web UI (HTTP API)

A minimal demo app for text-to-speech (TTS) workflows using:

- **Backend**: Python, `aiohttp`
- **TTS engines**: pluggable engines (currently `edge-tts` and `voicevox`)
- **Frontend**: plain HTML/JS
- **API transport**: HTTP (`GET`/`POST`)

## Features

- Enter text in the browser
- Send synthesis/stop/history commands to a **single endpoint** (`POST /api/command`)
- Select a TTS engine per request (`edge-tts` / `voicevox`)
- Long input is automatically split into manageable chunks, synthesized per chunk, and merged into a single audio file
- Synthesized audio is served as raw binary (`GET /api/history/{record_id}/audio`)
- Frontend can play generated audio and download it
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
python app.py --host 0.0.0.0 --port 9000 --voicevox-engine-url http://127.0.0.1:50021
```

- `--host`: host/interface to bind (default `0.0.0.0`)
- `--port`: port to bind (default `8080`)
- `--no-frontend`: do not serve `/` and `/frontend/*`; only backend endpoints are available
- `--voicevox-engine-url`: VoiceVox engine base URL (default `http://127.0.0.1:50021`)

## API

### Unified command endpoint

`POST /api/command`

#### Start synthesis (edge-tts example)

```json
{
  "action": "synthesize",
  "engine": "edge-tts",
  "text": "Hello from TTS Web UI",
  "voice": "en-US-JennyNeural",
  "rate": "+0%",
  "pitch": "+0Hz"
}
```

#### Start synthesis (voicevox example)

```json
{
  "action": "synthesize",
  "engine": "voicevox",
  "text": "こんにちは",
  "voice": "3"
}
```

#### List VoiceVox speakers

```json
{
  "action": "voicevox_speakers"
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

- Returns generated audio as raw binary (content type depends on engine and output format).
