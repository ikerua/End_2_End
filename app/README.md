# Whisper Eus — Interfaz Web Local

Aplicación web minimalista para transcripción en **euskera** usando tu modelo Whisper fine-tuneado. Backend en FastAPI, frontend en HTML/CSS/JS puro. Ejecutable localmente con Docker (CPU o GPU).

## Requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y en ejecución
- El modelo en `../modelo_base_hf/` (ya existe en tu proyecto)
- *(Opcional, para GPU)* Drivers NVIDIA + [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Uso rápido

Desde la carpeta `app2/`:

```bash
#── CPU (siempre funciona) ─────────────────────────────────────────────
docker compose up whisper-cpu

# Reconstruir imagen tras cambios de código:
docker compose up whisper-cpu --build

# ── GPU NVIDIA (más rápido, requiere nvidia-container-toolkit) ─────────
docker compose up whisper-gpu
```

Abre el navegador en: **http://localhost:7860**

## Parar el servidor

```bash
docker compose down
```

## Estructura

```
app2/
├── backend/
│   ├── server.py          # FastAPI + Whisper pipeline
│   └── requirements.txt
├── frontend/
│   ├── index.html         # Interfaz web
│   ├── style.css          # Dark theme minimalista
│   └── app.js             # Lógica de grabación y transcripción
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Endpoints del API

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Interfaz web |
| `GET` | `/status` | Estado del modelo (`loading \| loaded \| error`) |
| `POST` | `/transcribe` | Transcribir audio (form-data: `audio`, `use_chunking`, `chunk_length`) |

## Notas

- El modelo **no se incluye en la imagen Docker** — se monta como volumen desde `../modelo_base_hf/`.
- La grabación de micrófono requiere `localhost` o HTTPS (Docker usa `localhost`, correcto).
- Primera carga puede tardar 30-60s mientras el modelo se carga en memoria.
