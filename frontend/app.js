const textInput = document.getElementById("textInput");
const inputFormat = document.getElementById("inputFormat");
const speakerSelect = document.getElementById("speakerSelect");
const maxChunkChars = document.getElementById("maxChunkChars");
const synthesizeButton = document.getElementById("synthesizeButton");
const downloadButton = document.getElementById("downloadButton");
const audioPlayer = document.getElementById("audioPlayer");
const statusEl = document.getElementById("status");

let latestBlob = null;

async function loadSpeakers() {
  try {
    const res = await fetch("/api/speakers");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const speakers = await res.json();

    const styles = speakers.flatMap((speaker) =>
      speaker.styles.map((style) => ({
        id: style.id,
        label: `${speaker.name} - ${style.name}`,
      }))
    );

    speakerSelect.innerHTML = "";
    for (const style of styles) {
      const option = document.createElement("option");
      option.value = String(style.id);
      option.textContent = style.label;
      speakerSelect.appendChild(option);
    }
  } catch (err) {
    statusEl.textContent = `Failed to load speakers: ${err.message}`;
  }
}

function setStatus(message) {
  statusEl.textContent = message;
}

synthesizeButton.addEventListener("click", async () => {
  const text = textInput.value;
  if (!text.trim()) {
    setStatus("Please enter some text first.");
    return;
  }

  synthesizeButton.disabled = true;
  downloadButton.disabled = true;
  setStatus("Synthesizing...");

  try {
    const payload = {
      text,
      input_format: inputFormat.value,
      speaker: Number(speakerSelect.value || 1),
      max_chunk_chars: Number(maxChunkChars.value || 120),
    };

    const res = await fetch("/api/synthesize", {
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

loadSpeakers();
