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
const refreshHistoryButton = document.getElementById("refreshHistoryButton");
const historyList = document.getElementById("historyList");
const audioPlayer = document.getElementById("audioPlayer");
const statusEl = document.getElementById("status");

const STORAGE_KEY = "voicevox-webui-form";

let latestBlob = null;

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

function setStatus(message) {
  statusEl.textContent = message;
}

function renderHistory(items) {
  historyList.innerHTML = "";
  if (!Array.isArray(items) || items.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No synthesis history yet.";
    historyList.appendChild(li);
    return;
  }

  for (const item of items) {
    const li = document.createElement("li");
    const topLine = document.createElement("div");
    const createdAt = new Date(item.created_at).toLocaleString();
    topLine.textContent = `${createdAt} | speaker=${item.speaker} | chunks=${item.chunk_count}`;

    const textLine = document.createElement("div");
    textLine.className = "history-text";
    textLine.textContent = item.text;

    const cfgLine = document.createElement("div");
    cfgLine.className = "history-meta";
    cfgLine.textContent = `format=${item.input_format}, max_chunk_chars=${item.max_chunk_chars}, speed=${item.speed_scale}, pitch=${item.pitch_scale}`;

    const playButton = document.createElement("button");
    playButton.type = "button";
    playButton.textContent = "Play";
    playButton.addEventListener("click", async () => {
      try {
        setStatus("Loading history audio...");
        const res = await fetch(getApiUrl(`/api/history/${item.id}/audio`));
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const blob = await res.blob();
        latestBlob = blob;
        audioPlayer.src = URL.createObjectURL(blob);
        audioPlayer.play();
        downloadButton.disabled = false;
        setStatus("Loaded history audio.");
      } catch (err) {
        setStatus(`Failed to load history audio: ${err.message}`);
      }
    });

    li.appendChild(topLine);
    li.appendChild(textLine);
    li.appendChild(cfgLine);
    li.appendChild(playButton);
    historyList.appendChild(li);
  }
}

async function loadHistory() {
  try {
    const res = await fetch(getApiUrl("/api/history"));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderHistory(data.items || []);
  } catch (err) {
    setStatus(`Failed to load history: ${err.message}`);
  }
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

refreshHistoryButton.addEventListener("click", async () => {
  setStatus("Loading history...");
  await loadHistory();
  if (!statusEl.textContent.startsWith("Failed")) {
    setStatus("History updated.");
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

    const chunkCount = res.headers.get("X-Chunk-Count") || "?";
    const normalizedLen = res.headers.get("X-Normalized-Length") || "?";
    setStatus(`Done. Chunks: ${chunkCount}, normalized length: ${normalizedLen} chars.`);
    downloadButton.disabled = false;
    await loadHistory();
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

loadSavedInputs();
loadSpeakers();
loadHistory();
