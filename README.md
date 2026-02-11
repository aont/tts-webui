# VOICEVOX Web UI

A simple static HTML + JavaScript frontend with a Python backend that connects to a running `voicevox/voicevox_engine` Docker API.

## Features

- Enter plain text, Markdown, or HTML.
- Markdown/HTML are converted to plain text before synthesis.
- Long input is split into chunks and synthesized sequentially.
- Chunk audio is concatenated into a single WAV.
- Playback and download in the browser.

## Requirements

- Python 3.10+
- A running VOICEVOX engine API (for example, `http://localhost:50021`)

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open <http://localhost:8000>.

## Environment variables

- `VOICEVOX_BASE_URL` (default: `http://localhost:50021`)
- `MAX_CHARS_PER_CHUNK` (default: `180`)
- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `8000`)
