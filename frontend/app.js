const textInput = document.getElementById("textInput");
const backendEndpoint = document.getElementById("backendEndpoint");
const inputFormat = document.getElementById("inputFormat");
const speakerSelect = document.getElementById("speakerSelect");
const maxChunkChars = document.getElementById("maxChunkChars");
const speedScale = document.getElementById("speedScale");
const pitchScale = document.getElementById("pitchScale");
const synthesizeButton = document.getElementById("synthesizeButton");
const reloadSpeakersButton = document.getElementById("reloadSpeakersButton");
const downloadButton = document.getElementById("downloadButton");
const clearHistoryButton = document.getElementById("clearHistoryButton");
const audioPlayer = document.getElementById("audioPlayer");
const historyList = document.getElementById("historyList");
const statusEl = document.getElementById("status");

const STORAGE_KEY = "voicevox-webui-form";
const HISTORY_STORAGE_KEY = "voicevox-webui-history";
const MAX_HISTORY_ITEMS = 20;

let latestBlob = null;

function bytesToBase64(bytes) {
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

function base64ToBytes(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

async function blobToBase64(blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  return bytesToBase64(bytes);
}

function base64ToBlob(base64, mimeType = "audio/ogg") {
  const bytes = base64ToBytes(base64);
  return new Blob([bytes], { type: mimeType });
}

function readHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed;
  } catch {
    return [];
  }
}

function writeHistory(items) {
  localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(items));
}

function applyHistoryItem(item) {
  textInput.value = item.text || "";
  inputFormat.value = item.inputFormat || "auto";
  maxChunkChars.value = String(item.maxChunkChars || 120);
  speedScale.value = String(item.speedScale ?? 1.0);
  pitchScale.value = String(item.pitchScale ?? 0.0);
  if (item.speaker && Array.from(speakerSelect.options).some((opt) => opt.value === String(item.speaker))) {
    speakerSelect.value = String(item.speaker);
  }

  if (item.audioBase64) {
    latestBlob = base64ToBlob(item.audioBase64, item.audioType || "audio/ogg");
    audioPlayer.src = URL.createObjectURL(latestBlob);
  }

  saveInputs();
  setStatus("Loaded selected history item.");
}

function renderHistory() {
  const items = readHistory();
  historyList.innerHTML = "";

  if (!items.length) {
    historyList.textContent = "No history yet.";
    return;
  }

  for (const item of items) {
    const wrapper = document.createElement("article");
    wrapper.className = "history-item";

    const header = document.createElement("div");
    header.className = "history-header";

    const title = document.createElement("strong");
    title.textContent = item.label || "Synthesized audio";

    const when = document.createElement("small");
    when.textContent = new Date(item.createdAt || Date.now()).toLocaleString();

    header.append(title, when);

    const text = document.createElement("div");
    text.className = "history-text";
    text.textContent = item.text || "";

    const settings = document.createElement("div");
    settings.className = "history-settings";
    settings.textContent = `speaker: ${item.speaker}, format: ${item.inputFormat}, maxChunk: ${item.maxChunkChars}, speed: ${item.speedScale}, pitch: ${item.pitchScale}`;

    const historyAudio = document.createElement("audio");
    historyAudio.controls = true;
    if (item.audioBase64) {
      historyAudio.src = URL.createObjectURL(base64ToBlob(item.audioBase64, item.audioType || "audio/ogg"));
    }

    const actions = document.createElement("div");
    actions.className = "history-actions";

    const useButton = document.createElement("button");
    useButton.textContent = "Load this";
    useButton.addEventListener("click", () => applyHistoryItem(item));

    actions.appendChild(useButton);
    wrapper.append(header, text, settings, historyAudio, actions);
    historyList.appendChild(wrapper);
  }
}

async function addHistoryItem(payload, audioBlob) {
  const audioBase64 = await blobToBase64(audioBlob);
  const item = {
    id: String(Date.now()) + Math.random().toString(16).slice(2),
    createdAt: new Date().toISOString(),
    label: (payload.text || "").slice(0, 24) || "Synthesized audio",
    text: payload.text,
    inputFormat: payload.input_format,
    speaker: payload.speaker,
    maxChunkChars: payload.max_chunk_chars,
    speedScale: payload.speed_scale,
    pitchScale: payload.pitch_scale,
    audioType: audioBlob.type || "audio/ogg",
    audioBase64,
  };

  const current = readHistory();
  current.unshift(item);
  writeHistory(current.slice(0, MAX_HISTORY_ITEMS));
  renderHistory();
}

function normalizeEndpoint(value) {
  const trimmed = value.trim();
  if (!trimmed) return "";
  return trimmed.replace(/\/+$/, "");
}

function getApiUrl(path) {
  const endpoint = normalizeEndpoint(backendEndpoint.value);
  if (!endpoint) return path;
  return `${endpoint}${path}`;
}

function loadSavedInputs() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return;

  try {
    const parsed = JSON.parse(raw);
    if (typeof parsed.text === "string") textInput.value = parsed.text;
    if (typeof parsed.backendEndpoint === "string") backendEndpoint.value = parsed.backendEndpoint;
    if (typeof parsed.inputFormat === "string") inputFormat.value = parsed.inputFormat;
    if (typeof parsed.maxChunkChars === "string" || typeof parsed.maxChunkChars === "number") {
      maxChunkChars.value = String(parsed.maxChunkChars);
    }
    if (typeof parsed.speedScale === "string" || typeof parsed.speedScale === "number") {
      speedScale.value = String(parsed.speedScale);
    }
    if (typeof parsed.pitchScale === "string" || typeof parsed.pitchScale === "number") {
      pitchScale.value = String(parsed.pitchScale);
    }
  } catch {
    // ignore invalid localStorage content
  }
}

function saveInputs() {
  const payload = {
    text: textInput.value,
    backendEndpoint: backendEndpoint.value,
    inputFormat: inputFormat.value,
    speaker: speakerSelect.value,
    maxChunkChars: maxChunkChars.value,
    speedScale: speedScale.value,
    pitchScale: pitchScale.value,
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

async function loadSpeakers() {
  try {
    const res = await fetch(getApiUrl("/api/speakers"));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const speakers = await res.json();

    const styles = speakers.flatMap((speaker) =>
      speaker.styles.map((style) => ({
        id: style.id,
        label: `${speaker.name} - ${style.name}`,
      }))
    );

    const savedSpeaker = (() => {
      try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return "";
        const parsed = JSON.parse(raw);
        return typeof parsed.speaker === "string" ? parsed.speaker : "";
      } catch {
        return "";
      }
    })();

    speakerSelect.innerHTML = "";
    for (const style of styles) {
      const option = document.createElement("option");
      option.value = String(style.id);
      option.textContent = style.label;
      speakerSelect.appendChild(option);
    }

    if (savedSpeaker && styles.some((style) => String(style.id) === savedSpeaker)) {
      speakerSelect.value = savedSpeaker;
    }

    saveInputs();
  } catch (err) {
    statusEl.textContent = `Failed to load speakers: ${err.message}`;
  }
}

function setStatus(message) {
  statusEl.textContent = message;
}

[textInput, backendEndpoint, inputFormat, speakerSelect, maxChunkChars, speedScale, pitchScale].forEach((el) => {
  el.addEventListener("input", saveInputs);
  el.addEventListener("change", saveInputs);
});

reloadSpeakersButton.addEventListener("click", async () => {
  setStatus("Loading speakers...");
  await loadSpeakers();
  if (!statusEl.textContent.startsWith("Failed")) {
    setStatus("Speakers updated.");
  }
});

synthesizeButton.addEventListener("click", async () => {
  const text = textInput.value;
  if (!text.trim()) {
    setStatus("Please enter some text first.");
    return;
  }

  saveInputs();
  synthesizeButton.disabled = true;
  downloadButton.disabled = true;
  setStatus("Synthesizing...");

  try {
    const payload = {
      text,
      input_format: inputFormat.value,
      speaker: Number(speakerSelect.value || 1),
      max_chunk_chars: Number(maxChunkChars.value || 120),
      speed_scale: Number(speedScale.value || 1.0),
      pitch_scale: Number(pitchScale.value || 0.0),
    };

    const res = await fetch(getApiUrl("/api/synthesize"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      let detail = "";
      try {
        const json = await res.json();
        detail = json.error || JSON.stringify(json);
      } catch {
        detail = await res.text();
      }
      throw new Error(`Synthesis failed: HTTP ${res.status} ${detail}`);
    }

    latestBlob = await res.blob();
    const audioUrl = URL.createObjectURL(latestBlob);
    audioPlayer.src = audioUrl;
    await addHistoryItem(payload, latestBlob);

    const chunkCount = res.headers.get("X-Chunk-Count") || "?";
    const normalizedLen = res.headers.get("X-Normalized-Length") || "?";
    setStatus(`Done. Chunks: ${chunkCount}, normalized length: ${normalizedLen} chars.`);
    downloadButton.disabled = false;
  } catch (err) {
    setStatus(err.message);
  } finally {
    synthesizeButton.disabled = false;
  }
});

downloadButton.addEventListener("click", () => {
  if (!latestBlob) {
    setStatus("No audio generated yet.");
    return;
  }

  const url = URL.createObjectURL(latestBlob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "voicevox-output.opus";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});

clearHistoryButton.addEventListener("click", () => {
  writeHistory([]);
  renderHistory();
  setStatus("History cleared.");
});

loadSavedInputs();
loadSpeakers();
renderHistory();
