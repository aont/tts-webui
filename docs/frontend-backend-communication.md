# Frontend / Backend Communication Investigation

## Backend URL used by the frontend

- The frontend normalizes the base URL from `endpointInput` with `normalizeEndpointUrl()`.
- During normalization, it preserves both `origin` and `pathname` (and trims trailing `/`).
  - Example: if `https://host/path` is specified, API requests are sent to `https://host/path/...`.
- `getApiUrl(path)` builds URLs as `baseUrl + "/" + path`.

## Frontend → Backend endpoints and payloads

### 1) Start synthesis
- **URL**: `POST /synthesize`
- **JSON body**:
  - `text`: input text (after Markdown/HTML normalization)
  - `segments`: split text segments
  - `engine`: `edge-tts` / `voicevox` / `pyaitalk`
  - `voice`: engine-specific voice/style value
  - `style`: style name (VoiceVox only)
  - `rate`, `pitch`

### 2) List VoiceVox speakers
- **URL**: `GET /voicevox/speakers`

### 3) List pyaitalk voices
- **URL**: `GET /pyaitalk/voices`

### 4) List history
- **URL**: `GET /history`

### 5) Stop synthesis
- **URL**: `POST /stop`
- **JSON body**: `{ "record_id": "..." }`

### 6) Fetch generated audio file
- **URL**: `GET /history/{record_id}/audio`
- **Response**: binary audio (`audio/mpeg` or `audio/ogg`)

## Backend route surface

- Routes:
  - `POST /synthesize`
  - `GET /voicevox/speakers`
  - `GET /pyaitalk/voices`
  - `POST /stop`
  - `GET /history`
  - `GET /history/{record_id}/audio`
- The old action-dispatch API (`POST /api/command`) was removed and replaced with per-feature routes.

## Backend → external TTS engine calls

- VoiceVox:
  - `POST {voicevox_engine_url}/audio_query` (`text`, `speaker`)
  - `POST {voicevox_engine_url}/synthesis` (`speaker`, body=audio_query)
  - `GET {voicevox_engine_url}/speakers`
- pyaitalk:
  - `POST {pyaitalk_api_url}/synthesize` (`text`, `output=wav`, `character`)
  - `GET {pyaitalk_api_url}/voice/list`

Default values:
- `voicevox_engine_url = http://127.0.0.1:50021`
- `pyaitalk_api_url = http://127.0.0.1:8080`
