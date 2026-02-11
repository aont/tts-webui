# Edge TTS Web UI (WebSocket)

A minimal demo app using:

- **Backend**: Python, `aiohttp`, `edge-tts`
- **Frontend**: plain HTML/JS
- **Realtime channel**: **WebSocket only** via `/ws`

## Features

- Enter text in the browser
- Send synthesis requests to backend over `/ws`
- Backend generates speech with `edge-tts`
- Frontend can play the returned audio and download it as an MP3

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open: <http://localhost:8080>

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
