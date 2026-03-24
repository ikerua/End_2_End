"""
Whisper Eus — Real-Time Transcription Router
WebSocket endpoint that accepts binary audio chunks and returns partial transcriptions.
Designed to be imported and mounted into the main FastAPI app.
"""

import json
import time
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Shared state and helpers are imported from the main server module at runtime
# to avoid circular imports. We access them via a late import inside the handler.

realtime_router = APIRouter()


@realtime_router.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint for real-time transcription.

    Protocol (client → server):
        - binary frames: raw audio bytes (any format ffmpeg can decode)
        - text frame {"type": "stop"}: graceful shutdown

    Protocol (server → client):
        - {"type": "ready"}                         — after handshake
        - {"type": "transcript", "text": "...", "elapsed": 1.23}  — per chunk
        - {"type": "warning", "message": "..."}     — low-level audio, etc.
        - {"type": "error",   "message": "..."}     — model/ffmpeg failure
        - {"type": "stopped"}                        — after stop signal
    """
    # Late import to share state with main server module.
    # Works whether started as `python -m uvicorn backend.server:app` (package)
    # or from within the backend directory directly.
    try:
        from backend.server import state, load_audio_as_float32
    except ModuleNotFoundError:
        from server import state, load_audio_as_float32  # type: ignore
    import numpy as np

    await websocket.accept()

    # Check model is ready
    if state["status"] != "loaded":
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Modelo ez dago prest (egoera: {state['status']}). "
                       f"Errorea: {state.get('error', 'Ezezaguna')}",
        }))
        await websocket.close()
        return

    await websocket.send_text(json.dumps({"type": "ready"}))

    try:
        while True:
            # Accept either binary (audio chunk) or text (control) messages
            message = await websocket.receive()

            # ── Control message ──────────────────────────────────────────────
            if "text" in message:
                try:
                    ctrl = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                if ctrl.get("type") == "stop":
                    await websocket.send_text(json.dumps({"type": "stopped"}))
                    break
                continue

            # ── Audio chunk ──────────────────────────────────────────────────
            audio_bytes = message.get("bytes")
            if not audio_bytes:
                continue

            if len(audio_bytes) < 512:   # ignore tiny/empty frames
                continue

            # Decode in thread pool to avoid blocking the event loop
            try:
                loop = asyncio.get_event_loop()
                data, sample_rate = await loop.run_in_executor(
                    None, load_audio_as_float32, audio_bytes
                )
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Audio deskodetzeko errorea: {e}",
                }))
                continue

            # Sanity check — skip near-silent frames
            level = float(np.abs(data).mean())
            if level < 0.0008:
                await websocket.send_text(json.dumps({
                    "type": "warning",
                    "message": "Audio maila oso baxua da.",
                }))
                continue

            # Transcribe in thread pool
            try:
                t0 = time.time()
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: state["pipeline"](
                        {"sampling_rate": sample_rate, "raw": data},
                        generate_kwargs={"language": "basque", "task": "transcribe"},
                    ),
                )
                elapsed = round(time.time() - t0, 2)

                text = result.get("text", "").strip()
                if not text or len(text) < 2:
                    continue

                await websocket.send_text(json.dumps({
                    "type": "transcript",
                    "text": text,
                    "elapsed": elapsed,
                }))

            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Transkripzio errorea: {e}",
                }))

    except WebSocketDisconnect:
        pass  # Client disconnected — nothing to do
