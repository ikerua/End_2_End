# Whisper Eus — Interfaz Web Local

Aplicación web para transcripción en **euskera** usando tu modelo Whisper fine-tuneado.
Backend en FastAPI, frontend en HTML/CSS/JS puro. Dos modos de despliegue Docker: **slim** (modelo en el host) y **bundled** (modelo incluido en la imagen).

---

## Modos de despliegue

### 🪶 Slim — modelo en el host (imagen ligera)

La imagen Docker **no incluye** el modelo. Se monta como volumen desde `../modelo_base_hf/`.

| Ventaja | Inconveniente |
|---|---|
| Imagen pequeña (~1 GB) | Requiere tener el modelo en el host |
| Reconstrucción rápida | No portable a otro PC sin el modelo |

```bash
# Desde la raíz del proyecto (End_2_End/)
docker compose up whisper-cpu-slim

# GPU NVIDIA
docker compose up whisper-gpu-slim
```

---

### 📦 Bundled — modelo incluido en la imagen

La imagen Docker **incluye el modelo** fine-tuneado. No necesita ningún volumen adicional.

| Ventaja | Inconveniente |
|---|---|
| Completamente autocontenida | Imagen grande (~2 GB+) |
| Portable a cualquier máquina con Docker | Build inicial más lento |

```bash
# Desde la raíz del proyecto (End_2_End/)
docker compose up whisper-cpu-bundled

# GPU NVIDIA
docker compose up whisper-gpu-bundled
```

---

## Requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y en ejecución
- *(Solo slim)* Modelo en `./modelo_base_hf/`
- *(Solo GPU)* Drivers NVIDIA + [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

> **Todos los comandos docker compose se ejecutan desde la raíz del proyecto** (`End_2_End/`).

## Acceder a la aplicación

Abre el navegador en: **http://localhost:7860**

## Parar el servidor

```bash
# Desde la raíz del proyecto
docker compose down
```

## Forzar reconstrucción

```bash
# Desde la raíz del proyecto
docker compose up whisper-cpu-slim --build
docker compose up whisper-cpu-bundled --build
```

---

## Logs de transcripción

Cada transcripción (archivo y tiempo real) queda registrada en `logs/transcription.jsonl`.
Los logs se montan desde el host en todos los modos, por lo que persisten entre reinicios.

```jsonc
{
  "timestamp":        "2026-03-25T13:05:56.000Z",
  "mode":             "file",     // "file" | "realtime"
  "device":           "CPU",      // "CPU"  | "CUDA"
  "elapsed_s":        1.74,
  "audio_duration_s": 4.51,
  "audio_bytes":      145300,
  "text_chars":       87,
  "text_words":       14,
  "audio_level":      0.0123
}
```

---

## Estructura

```
app/
├── backend/
│   ├── server.py            # FastAPI + Whisper pipeline
│   ├── server_realtime.py   # WebSocket tiempo real
│   ├── logger.py            # Logging de métricas
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   ├── realtime.css
│   └── realtime.js
├── logs/                    # Logs JSONL persistentes
├── Dockerfile               # Imagen slim
├── Dockerfile.bundled       # Imagen con modelo incluido
├── docker-compose.yml
└── README.md
```

## Endpoints del API

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Interfaz web |
| `GET` | `/status` | Estado del modelo (`loading \| loaded \| error`) |
| `POST` | `/transcribe` | Transcribir audio (form-data: `audio`, `use_chunking`, `chunk_length`) |
| `WS`  | `/ws/transcribe` | Transcripción en tiempo real (WebSocket) |
