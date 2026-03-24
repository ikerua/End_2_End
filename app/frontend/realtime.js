/* ── Whisper Eus — Real-Time Transcription Logic ─────────────────────────── */

// ── WebSocket & Recording State ───────────────────────────────────────────────
const rtState = {
  isStreaming: false,
  ws: null,
  mediaRecorder: null,
  audioStream: null,
  analyser: null,
  animFrame: null,
  chunkCount: 0,
};

// ── DOM Refs ──────────────────────────────────────────────────────────────────
const rt = {
  // Tabs
  tabRecord:       document.getElementById('tab-record'),
  tabRealtime:     document.getElementById('tab-realtime'),
  panelRecord:     document.getElementById('panel-record'),
  panelRealtime:   document.getElementById('panel-realtime'),
  // Real-time controls
  btnStream:       document.getElementById('btn-stream'),
  liveIndicator:   document.getElementById('live-indicator'),
  liveDot:         document.getElementById('live-dot'),
  liveLabel:       document.getElementById('live-label'),
  realtimeHint:    document.getElementById('realtime-hint'),
  realtimeWaveform:document.getElementById('realtime-waveform'),
  realtimeCanvas:  document.getElementById('realtime-canvas'),
  realtimePulse:   document.getElementById('realtime-pulse'),
};

// Watch the status badge so we enable btn-stream when model is ready
// (avoids touching app.js)
const statusBadge = document.getElementById('status-badge');
if (statusBadge) {
  const observer = new MutationObserver(() => {
    if (statusBadge.classList.contains('status-loaded')) {
      if (rt.btnStream) rt.btnStream.disabled = false;
    }
  });
  observer.observe(statusBadge, { attributes: true, attributeFilter: ['class'] });
}

const rtCtx = rt.realtimeCanvas?.getContext('2d');

// ── Tab Switching ──────────────────────────────────────────────────────────────
function activateTab(tab) {
  if (tab === 'record') {
    rt.tabRecord.classList.add('active');
    rt.tabRealtime.classList.remove('active');
    rt.panelRecord.classList.add('active');
    rt.panelRealtime.classList.remove('active');
  } else {
    rt.tabRealtime.classList.add('active');
    rt.tabRecord.classList.remove('active');
    rt.panelRealtime.classList.add('active');
    rt.panelRecord.classList.remove('active');
  }
}

rt.tabRecord?.addEventListener('click', () => {
  if (rtState.isStreaming) stopStreaming();
  activateTab('record');
});

rt.tabRealtime?.addEventListener('click', () => activateTab('realtime'));

// ── Real-Time Waveform ────────────────────────────────────────────────────────
function resizeRtCanvas() {
  if (!rt.realtimeCanvas || !rt.realtimeWaveform) return;
  rt.realtimeCanvas.width  = rt.realtimeWaveform.clientWidth;
  rt.realtimeCanvas.height = rt.realtimeWaveform.clientHeight;
}

window.addEventListener('resize', resizeRtCanvas);
resizeRtCanvas();

function drawRtIdle() {
  if (!rtCtx || !rt.realtimeCanvas) return;
  const { width, height } = rt.realtimeCanvas;
  rtCtx.clearRect(0, 0, width, height);
  const mid = height / 2;
  rtCtx.beginPath();
  rtCtx.moveTo(0, mid);
  rtCtx.lineTo(width, mid);
  rtCtx.strokeStyle = 'rgba(255,255,255,0.07)';
  rtCtx.lineWidth = 1;
  rtCtx.stroke();
}

function startRtWaveform(stream) {
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(stream);
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  source.connect(analyser);
  rtState.analyser = analyser;

  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function draw() {
    rtState.animFrame = requestAnimationFrame(draw);
    analyser.getByteTimeDomainData(dataArray);

    if (!rt.realtimeCanvas) return;
    const { width, height } = rt.realtimeCanvas;
    rtCtx.clearRect(0, 0, width, height);

    const gradient = rtCtx.createLinearGradient(0, 0, width, 0);
    gradient.addColorStop(0,   'rgba(34, 197, 94, 0.15)');
    gradient.addColorStop(0.5, 'rgba(34, 197, 94, 0.8)');
    gradient.addColorStop(1,   'rgba(34, 197, 94, 0.15)');

    rtCtx.lineWidth = 2;
    rtCtx.strokeStyle = gradient;
    rtCtx.beginPath();

    const sliceWidth = width / bufferLength;
    let x = 0;
    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * height) / 2;
      if (i === 0) rtCtx.moveTo(x, y);
      else         rtCtx.lineTo(x, y);
      x += sliceWidth;
    }
    rtCtx.lineTo(width, height / 2);
    rtCtx.stroke();
  }

  draw();
}

function stopRtWaveform() {
  if (rtState.animFrame) {
    cancelAnimationFrame(rtState.animFrame);
    rtState.animFrame = null;
  }
  rtState.analyser = null;
  drawRtIdle();
}

// ── Streaming ─────────────────────────────────────────────────────────────────
async function startStreaming() {
  // Connect WebSocket first
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws/transcribe`);
  rtState.ws = ws;
  rtState.liveEntry   = null;   // the single growing DOM card
  rtState.liveText    = '';     // accumulated text

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      switch (msg.type) {
        case 'ready':
          beginCapture(ws);
          break;

        case 'transcript': {
          // Append to accumulated text
          rtState.liveText += (rtState.liveText ? ' ' : '') + msg.text;

          if (!rtState.liveEntry) {
            // Create the live card once
            rtState.liveEntry = createLiveEntry();
          }

          // Update the text in-place
          const textEl = rtState.liveEntry.querySelector('.transcript-text');
          if (textEl) textEl.textContent = rtState.liveText;

          // Scroll to bottom
          const scroll = document.getElementById('transcript-scroll');
          if (scroll) scroll.scrollTo({ top: scroll.scrollHeight, behavior: 'smooth' });

          rtState.chunkCount++;
          rt.realtimeHint.textContent = `${rtState.chunkCount} zati transkribatu`;
          rt.realtimeHint.className = 'realtime-hint streaming';
          break;
        }

        case 'warning':
          showError(msg.message);
          setTimeout(hideError, 3000);
          break;

        case 'error':
          showError(msg.message);
          stopStreaming();
          break;

        case 'stopped':
          ws.close();
          break;
      }
    } catch (_) { /* ignore parse errors */ }
  };

  ws.onerror = () => {
    showError('WebSocket konexio errorea.');
    stopStreaming();
  };

  ws.onclose = () => {
    rtState.ws = null;
  };
}

/** Creates and inserts the live transcript card into the list. */
function createLiveEntry() {
  const transcriptEmpty = document.getElementById('transcript-empty');
  const transcriptList  = document.getElementById('transcript-list');
  const btnCopy  = document.getElementById('btn-copy');
  const btnClear = document.getElementById('btn-clear');

  if (transcriptEmpty) transcriptEmpty.style.display = 'none';
  if (btnCopy)  btnCopy.disabled  = false;
  if (btnClear) btnClear.disabled = false;

  const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

  const entry = document.createElement('div');
  entry.className = 'transcript-entry live-entry';
  entry.innerHTML = `
    <div class="transcript-meta">
      <span class="transcript-tag realtime">📡 Zuzenean</span>
      <div style="display:flex; gap:12px; align-items:center;">
        <span class="live-badge">● LIVE</span>
        <span>${timeStr}</span>
      </div>
    </div>
    <p class="transcript-text"></p>
  `;

  transcriptList?.appendChild(entry);
  return entry;
}


async function beginCapture(ws) {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    rtState.audioStream = stream;
    rtState.isStreaming = true;
    rtState.chunkCount = 0;

    startRtWaveform(stream);
    rt.realtimeWaveform?.classList.add('active');

    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : 'audio/webm';

    // ── Stop/restart pattern ─────────────────────────────────────────────────
    // Each recorder session produces a complete WebM blob (header + data),
    // so ffmpeg can always decode it — avoids the "Invalid data" error from
    // sending headerless timeslice chunks.
    let pendingChunks = [];

    function createRecorder() {
      const mr = new MediaRecorder(stream, { mimeType });
      rtState.mediaRecorder = mr;

      mr.ondataavailable = (e) => {
        if (e.data.size > 0) pendingChunks.push(e.data);
      };

      mr.onstop = () => {
        // Send complete blob (always has WebM header)
        if (pendingChunks.length > 0 && ws.readyState === WebSocket.OPEN) {
          const blob = new Blob(pendingChunks, { type: mimeType });
          pendingChunks = [];
          ws.send(blob);
        } else {
          pendingChunks = [];
        }
        // Chain: restart immediately if still streaming
        if (rtState.isStreaming) createRecorder();
      };

      mr.start();
    }

    createRecorder();

    // Trigger a new session every 3 s
    rtState.chunkInterval = setInterval(() => {
      if (rtState.isStreaming &&
          rtState.mediaRecorder &&
          rtState.mediaRecorder.state === 'recording') {
        rtState.mediaRecorder.stop(); // onstop will restart
      }
    }, 3000);

    // Update UI
    rt.btnStream.classList.add('streaming');
    rt.realtimePulse?.classList.add('active');
    rt.liveIndicator.classList.add('live');
    rt.liveLabel.textContent = 'Zuzen';
    rt.realtimeHint.textContent = 'Entzuten...';
    rt.realtimeHint.className = 'realtime-hint streaming';

    // Disable tab switching while streaming
    rt.tabRecord.disabled = true;
    rt.tabRecord.style.opacity = '0.4';
    rt.tabRecord.style.cursor = 'not-allowed';
  } catch (err) {
    showError('Ezin da mikrofonoa atzitu: ' + err.message);
    stopStreaming();
  }
}

function stopStreaming() {
  rtState.isStreaming = false; // set first so onstop doesn't restart

  // Finalize the live entry: remove LIVE badge, register text for copy
  if (rtState.liveEntry && rtState.liveText) {
    const badge = rtState.liveEntry.querySelector('.live-badge');
    if (badge) badge.remove();
    // Push to the shared transcriptions array so btn-copy works
    state.transcriptions.push({
      text: rtState.liveText,
      source: 'realtime',
      elapsed: null,
      time: new Date(),
    });
  }
  rtState.liveEntry = null;
  rtState.liveText  = '';

  if (rtState.chunkInterval) {
    clearInterval(rtState.chunkInterval);
    rtState.chunkInterval = null;
  }

  if (rtState.mediaRecorder && rtState.mediaRecorder.state !== 'inactive') {
    rtState.mediaRecorder.stop();
  }

  if (rtState.audioStream) {
    rtState.audioStream.getTracks().forEach(t => t.stop());
    rtState.audioStream = null;
  }

  stopRtWaveform();
  rt.realtimeWaveform?.classList.remove('active');

  if (rtState.ws && rtState.ws.readyState === WebSocket.OPEN) {
    rtState.ws.send(JSON.stringify({ type: 'stop' }));
  }

  rtState.mediaRecorder = null;

  // Reset UI
  rt.btnStream.classList.remove('streaming');
  rt.realtimePulse?.classList.remove('active');
  rt.liveIndicator.classList.remove('live');
  rt.liveLabel.textContent = 'Gelditurik';
  rt.realtimeHint.textContent = 'Sakatu zuzenean transkribatzeko';
  rt.realtimeHint.className = 'realtime-hint';

  // Re-enable tab switching
  rt.tabRecord.disabled = false;
  rt.tabRecord.style.opacity = '';
  rt.tabRecord.style.cursor = '';
}

// ── Stream button ──────────────────────────────────────────────────────────────
rt.btnStream?.addEventListener('click', () => {
  if (!state.modelLoaded) return;
  if (rtState.isStreaming) {
    stopStreaming();
  } else {
    hideError();
    startStreaming();
  }
});

// ── Init ──────────────────────────────────────────────────────────────────────
drawRtIdle();
