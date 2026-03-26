"""
Whisper Eus — Transcription Logger
Registra métricas de cada transcripción en un archivo de log JSONL y en consola.

Campos registrados:
  - timestamp        : ISO-8601 del momento de la transcripción
  - mode             : "file" | "realtime"
  - device           : "cpu" | "cuda"
  - elapsed_s        : tiempo de transcripción en segundos
  - audio_duration_s : duración estimada del audio en segundos (samples / 16000)
  - audio_samples    : número de muestras float32
  - audio_bytes      : tamaño del archivo de audio en bytes
  - text_chars       : número de caracteres en el resultado
  - text_words       : número de palabras en el resultado
  - audio_level      : nivel RMS promedio del audio
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

# ── Log file location ──────────────────────────────────────────────────────────
# Defaults to  <project_root>/logs/transcription.jsonl
# Override with env var TRANSCRIPTION_LOG_PATH
_DEFAULT_LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_PATH = Path(os.environ.get("TRANSCRIPTION_LOG_PATH",
                               str(_DEFAULT_LOG_DIR / "transcription.jsonl")))

# ── Console logger (structred line per request) ────────────────────────────────
_console = logging.getLogger("whisper_eus.transcription")
if not _console.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    _console.addHandler(_handler)
    _console.setLevel(logging.INFO)


def log_transcription(
    *,
    mode: str,           # "file" | "realtime"
    device: str,         # "cpu"  | "cuda"
    elapsed_s: float,
    audio_samples: int,
    audio_bytes: int,
    text: str,
    audio_level: float,
    sample_rate: int = 16_000,
) -> None:
    """
    Persiste una entrada de log en JSONL y emite una línea formateada a la consola.
    Nunca lanza excepciones: si falla, lo anota en consola y continúa.
    """
    audio_duration_s = round(audio_samples / sample_rate, 3)
    entry = {
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "mode":             mode,
        "device":           device.upper(),
        "elapsed_s":        elapsed_s,
        "audio_duration_s": audio_duration_s,
        "audio_samples":    audio_samples,
        "audio_bytes":      audio_bytes,
        "text_chars":       len(text),
        "text_words":       len(text.split()) if text else 0,
        "audio_level":      round(audio_level, 6),
    }

    # ── Console ────────────────────────────────────────────────────────────────
    rtf = round(elapsed_s / audio_duration_s, 3) if audio_duration_s > 0 else None
    _console.info(
        "[%s | %s] %.2fs transcripción / %.2fs audio (RTF %.3f) "
        "| %d chars / %d palabras | nivel=%.4f",
        mode.upper(), device.upper(),
        elapsed_s, audio_duration_s, rtf or 0,
        entry["text_chars"], entry["text_words"], audio_level,
    )

    # ── JSONL file ─────────────────────────────────────────────────────────────
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001
        _console.warning("No se pudo escribir en el log JSONL: %s", exc)
