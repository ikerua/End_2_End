# ── Whisper Eus — Dockerfile ──────────────────────────────────────────────────
# Supports both CPU and GPU (CUDA) via docker-compose profiles.
# The model is NOT baked into the image; it is mounted as a volume.

FROM python:3.12-slim

# System dependencies (ffmpeg for audio decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend and frontend
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
  CMD curl -f http://localhost:7860/status || exit 1

# Run server
CMD ["python", "-m", "uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
