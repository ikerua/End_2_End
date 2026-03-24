"""
Whisper Eus — Backend Server
FastAPI server for local Whisper transcription (Basque).
Auto-detects CUDA GPU or falls back to CPU.
"""

import io
import os
import time
import tempfile
import subprocess
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import pipeline

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = os.environ.get("MODEL_PATH", str(BASE_DIR / "modelo"))
FRONTEND_DIR = BASE_DIR / "frontend"

# ── Model State ────────────────────────────────────────────────────────────────
state = {
    "status": "loading",  # loading | loaded | error
    "device": None,
    "error": None,
    "pipeline": None,
    "load_time": None,
}

# ── Lifespan: load model on startup ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("🚀 Cargando modelo Whisper...")
    print(f"   Ruta: {MODEL_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Dispositivo: {device.upper()}")
    state["device"] = device

    t0 = time.time()
    try:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Modelo no encontrado en '{MODEL_PATH}'. "
                "Asegúrate de que el volumen está montado correctamente."
            )

        state["pipeline"] = pipeline(
            "automatic-speech-recognition",
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,
            feature_extractor=MODEL_PATH,
            device=device,
        )
        state["status"] = "loaded"
        state["load_time"] = round(time.time() - t0, 2)
        print(f"✅ Modelo cargado en {state['load_time']}s ({device.upper()})\n")
    except Exception as e:
        state["status"] = "error"
        state["error"] = str(e)
        print(f"❌ Error cargando modelo: {e}\n")

    yield  # app runs here

    print("👋 Cerrando servidor...")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Whisper Eus", version="1.0.0", lifespan=lifespan)


def load_audio_as_float32(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Use ffmpeg to decode any audio format to 16kHz mono float32 PCM.
    Returns (numpy_array, sample_rate).
    """
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_in_path,
            "-ar", "16000",
            "-ac", "1",
            "-f", "f32le",
            "-"
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr.decode()}")

        raw = np.frombuffer(result.stdout, dtype=np.float32)
        return raw, 16000
    finally:
        os.unlink(tmp_in_path)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/status")
def get_status():
    return {
        "status": state["status"],
        "device": state["device"],
        "error": state["error"],
        "load_time": state["load_time"],
        "model_path": MODEL_PATH,
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    use_chunking: bool = Form(False),
    chunk_length: int = Form(15),
):
    if state["status"] != "loaded":
        raise HTTPException(
            status_code=503,
            detail=f"Modelo no disponible (estado: {state['status']}). "
                   f"Error: {state.get('error', 'Desconocido')}",
        )

    audio_bytes = await audio.read()

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Archivo de audio vacío.")

    # Decode audio
    try:
        data, sample_rate = load_audio_as_float32(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error decodificando audio: {e}")

    # Sanity check
    level = float(np.abs(data).mean())
    if level < 0.001:
        return JSONResponse({"text": "", "warning": "Nivel de audio muy bajo. ¿Está el micrófono silenciado?"})

    # Transcribe
    try:
        kwargs = {
            "return_timestamps": True,
            "generate_kwargs": {"language": "basque", "task": "transcribe"},
        }
        if use_chunking:
            kwargs["chunk_length_s"] = max(2, min(30, chunk_length))

        t0 = time.time()
        result = state["pipeline"](
            {"sampling_rate": sample_rate, "raw": data},
            **kwargs,
        )
        elapsed = round(time.time() - t0, 2)

        text = result.get("text", "").strip()

        if not text or len(text) < 2:
            return JSONResponse({"text": "", "warning": "No se detectó habla."})

        return JSONResponse({
            "text": text,
            "elapsed": elapsed,
            "device": state["device"],
            "audio_level": round(level, 4),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en transcripción: {e}")


# ── Static frontend ────────────────────────────────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
