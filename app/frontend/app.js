/* ── Whisper Eus — App Logic ────────────────────────────────────────────────── */

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  modelLoaded: false,
  isRecording: false,
  isProcessing: false,
  mediaRecorder: null,
  audioChunks: [],
  audioStream: null,
  analyser: null,
  animFrame: null,
  transcriptions: [],
};

// ── DOM Refs ──────────────────────────────────────────────────────────────────
const dom = {
  statusBadge:    document.getElementById('status-badge'),
  statusLabel:    document.getElementById('status-label'),
  statusDevice:   document.getElementById('status-device'),
  btnRecord:      document.getElementById('btn-record'),
  pulseRing:      document.getElementById('pulse-ring'),
  recordHint:     document.getElementById('record-hint'),
  waveformCanvas: document.getElementById('waveform-canvas'),
  waveform:       document.getElementById('waveform-container'),
  fileInput:      document.getElementById('file-input'),
  btnUpload:      document.getElementById('btn-upload'),
  fileName:       document.getElementById('file-name'),
  chkChunking:    document.getElementById('chk-chunking'),
  chunkSlider:    document.getElementById('chunk-slider'),
  chunkVal:       document.getElementById('chunk-val'),
  chunkContainer: document.getElementById('chunk-slider-container'),
  transcriptList: document.getElementById('transcript-list'),
  transcriptEmpty:document.getElementById('transcript-empty'),
  transcriptScroll:document.getElementById('transcript-scroll'),
  processingBar:  document.getElementById('processing-bar'),
  processingLabel:document.getElementById('processing-label'),
  errorBanner:    document.getElementById('error-banner'),
  errorText:      document.getElementById('error-text'),
  btnCopy:        document.getElementById('btn-copy'),
  btnClear:       document.getElementById('btn-clear'),
  footerDevice:   document.getElementById('footer-device'),
};

// ── Canvas setup ──────────────────────────────────────────────────────────────
const ctx2d = dom.waveformCanvas.getContext('2d');

function resizeCanvas() {
  dom.waveformCanvas.width  = dom.waveform.clientWidth;
  dom.waveformCanvas.height = dom.waveform.clientHeight;
}

window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// ── Status Polling ────────────────────────────────────────────────────────────
async function pollStatus() {
  try {
    const res = await fetch('/status');
    const data = await res.json();

    dom.statusBadge.className = `status-badge status-${data.status}`;

    if (data.status === 'loaded') {
      dom.statusLabel.textContent = 'Prest';
      dom.statusDevice.textContent = data.device?.toUpperCase() || '';
      dom.footerDevice.textContent = data.device ? `${data.device.toUpperCase()} · ${data.load_time}s kargatu` : '';
      dom.btnRecord.disabled = false;
      dom.btnUpload.disabled = false;
      dom.recordHint.textContent = 'Sakatu grabatzeko';
      state.modelLoaded = true;
    } else if (data.status === 'error') {
      dom.statusLabel.textContent = 'Errorea';
      dom.statusDevice.textContent = '';
      dom.recordHint.textContent = 'Errorea eredua kargatzean';
      showError(data.error || 'Ezezaguna');
    } else {
      dom.statusLabel.textContent = 'Kargatzen...';
      dom.recordHint.textContent = 'Eredua kargatzen ari da...';
      setTimeout(pollStatus, 2000);
    }
  } catch {
    setTimeout(pollStatus, 3000);
  }
}

// ── Waveform Visualization ────────────────────────────────────────────────────
function drawIdle() {
  const { width, height } = dom.waveformCanvas;
  ctx2d.clearRect(0, 0, width, height);
  const mid = height / 2;
  ctx2d.beginPath();
  ctx2d.moveTo(0, mid);
  ctx2d.lineTo(width, mid);
  ctx2d.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx2d.lineWidth = 1;
  ctx2d.stroke();
}

function startWaveform(stream) {
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(stream);
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  source.connect(analyser);
  state.analyser = analyser;

  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function draw() {
    state.animFrame = requestAnimationFrame(draw);
    analyser.getByteTimeDomainData(dataArray);

    const { width, height } = dom.waveformCanvas;
    ctx2d.clearRect(0, 0, width, height);

    const gradient = ctx2d.createLinearGradient(0, 0, width, 0);
    gradient.addColorStop(0,   'rgba(124, 58, 237, 0.15)');
    gradient.addColorStop(0.5, 'rgba(124, 58, 237, 0.8)');
    gradient.addColorStop(1,   'rgba(124, 58, 237, 0.15)');

    ctx2d.lineWidth = 2;
    ctx2d.strokeStyle = gradient;
    ctx2d.beginPath();

    const sliceWidth = width / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * height) / 2;
      if (i === 0) ctx2d.moveTo(x, y);
      else         ctx2d.lineTo(x, y);
      x += sliceWidth;
    }
    ctx2d.lineTo(width, height / 2);
    ctx2d.stroke();
  }

  draw();
}

function stopWaveform() {
  if (state.animFrame) {
    cancelAnimationFrame(state.animFrame);
    state.animFrame = null;
  }
  state.analyser = null;
  drawIdle();
}

// ── Recording ─────────────────────────────────────────────────────────────────
async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.audioStream = stream;

    startWaveform(stream);
    dom.waveform.classList.add('active');

    state.audioChunks = [];
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : 'audio/webm';

    const mediaRecorder = new MediaRecorder(stream, { mimeType });
    state.mediaRecorder = mediaRecorder;

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) state.audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      const blob = new Blob(state.audioChunks, { type: mimeType });
      state.audioStream.getTracks().forEach(t => t.stop());
      stopWaveform();
      dom.waveform.classList.remove('active');
      await sendAudio(blob, 'mic');
    };

    mediaRecorder.start();
    state.isRecording = true;

    dom.btnRecord.classList.add('recording');
    dom.pulseRing.classList.add('active');
    dom.recordHint.textContent = 'Gelditu grabatzeko...';
    dom.recordHint.className = 'record-hint recording';
    dom.btnUpload.disabled = true;
  } catch (err) {
    showError('Ezin da mikrofonoa atzitu: ' + err.message);
  }
}

function stopRecording() {
  if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
    state.mediaRecorder.stop();
  }
  state.isRecording = false;

  dom.btnRecord.classList.remove('recording');
  dom.pulseRing.classList.remove('active');
  dom.recordHint.textContent = 'Prozesatzen...';
  dom.recordHint.className = 'record-hint processing';
}

dom.btnRecord.addEventListener('click', () => {
  if (!state.modelLoaded || state.isProcessing) return;
  if (state.isRecording) stopRecording();
  else startRecording();
});

// ── File Upload ───────────────────────────────────────────────────────────────
dom.btnUpload.addEventListener('click', () => dom.fileInput.click());

dom.fileInput.addEventListener('change', async () => {
  const file = dom.fileInput.files[0];
  if (!file) return;
  dom.fileName.textContent = file.name;
  await sendAudio(file, 'file');
  dom.fileInput.value = '';
});

// ── Send Audio ────────────────────────────────────────────────────────────────
async function sendAudio(blobOrFile, source) {
  if (state.isProcessing) return;
  state.isProcessing = true;

  setProcessing(true, source === 'mic' ? '🎙️ Transkribatzen...' : '📁 Fitxategia prozesatzen...');
  hideError();

  const useChunking = dom.chkChunking.checked;
  const chunkLength = parseInt(dom.chunkSlider.value, 10);

  const ext   = source === 'mic' ? 'webm' : (blobOrFile.name?.split('.').pop() || 'audio');
  const name  = source === 'mic' ? `recording.${ext}` : blobOrFile.name;

  const formData = new FormData();
  formData.append('audio', blobOrFile, name);
  formData.append('use_chunking', useChunking);
  formData.append('chunk_length', chunkLength);

  try {
    const res = await fetch('/transcribe', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      showError(data.detail || `HTTP ${res.status}`);
      return;
    }

    if (data.warning) {
      showError(data.warning);
      return;
    }

    if (data.text) {
      addTranscription(data.text, source, data.elapsed);
    }
  } catch (err) {
    showError('Sare errorea: ' + err.message);
  } finally {
    state.isProcessing = false;
    setProcessing(false);
    dom.btnRecord.disabled = !state.modelLoaded;
    dom.btnUpload.disabled = !state.modelLoaded;
    dom.recordHint.textContent = 'Sakatu grabatzeko';
    dom.recordHint.className = 'record-hint';
    dom.fileName.textContent = '';
  }
}

// ── Transcription List ────────────────────────────────────────────────────────
function addTranscription(text, source, elapsed) {
  state.transcriptions.push({ text, source, elapsed, time: new Date() });

  dom.transcriptEmpty.style.display = 'none';
  dom.btnCopy.disabled  = false;
  dom.btnClear.disabled = false;

  const entry = document.createElement('div');
  entry.className = 'transcript-entry';

  const tagLabel = source === 'mic' ? '🎙 Micro' : '📁 Fitxategia';
  const tagClass = source === 'mic' ? '' : 'file';
  const timeStr  = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const elapsedStr = elapsed ? `${elapsed}s` : '';

  entry.innerHTML = `
    <div class="transcript-meta">
      <span class="transcript-tag ${tagClass}">${tagLabel}</span>
      <div style="display:flex; gap:12px; align-items:center;">
        ${elapsedStr ? `<span class="transcript-elapsed">⏱ ${elapsedStr}</span>` : ''}
        <span>${timeStr}</span>
      </div>
    </div>
    <p class="transcript-text">${escapeHTML(text)}</p>
  `;

  dom.transcriptList.appendChild(entry);
  dom.transcriptScroll.scrollTo({ top: dom.transcriptScroll.scrollHeight, behavior: 'smooth' });
}

function escapeHTML(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ── Copy / Clear ──────────────────────────────────────────────────────────────
dom.btnCopy.addEventListener('click', async () => {
  const allText = state.transcriptions.map(t => t.text).join('\n\n');
  await navigator.clipboard.writeText(allText);
  dom.btnCopy.classList.add('success');
  setTimeout(() => dom.btnCopy.classList.remove('success'), 1500);
});

dom.btnClear.addEventListener('click', () => {
  state.transcriptions = [];
  // Remove all transcript entries except empty state
  const entries = dom.transcriptList.querySelectorAll('.transcript-entry');
  entries.forEach(e => e.remove());
  dom.transcriptEmpty.style.display = '';
  dom.btnCopy.disabled  = true;
  dom.btnClear.disabled = true;
  hideError();
});

// ── Chunking controls ─────────────────────────────────────────────────────────
dom.chkChunking.addEventListener('change', () => {
  dom.chunkContainer.classList.toggle('hidden', !dom.chkChunking.checked);
});

dom.chunkSlider.addEventListener('input', () => {
  dom.chunkVal.textContent = `${dom.chunkSlider.value}s`;
});

// ── UI Helpers ────────────────────────────────────────────────────────────────
function setProcessing(active, label = '') {
  dom.processingBar.classList.toggle('hidden', !active);
  if (active) dom.processingLabel.textContent = label;
  dom.btnRecord.disabled = active || !state.modelLoaded;
  dom.btnUpload.disabled = active || !state.modelLoaded;
}

function showError(msg) {
  dom.errorBanner.classList.remove('hidden');
  dom.errorText.textContent = msg;
}

function hideError() {
  dom.errorBanner.classList.add('hidden');
}

// ── Init ──────────────────────────────────────────────────────────────────────
drawIdle();
pollStatus();
