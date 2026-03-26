# Hitzontzi — Transcripción Automática en Euskera

Sistema de transcripción automática de voz (ASR) para euskera, basado en un modelo [Whisper](https://github.com/openai/whisper) fine-tuned. Incluye una aplicación web para transcripción estándar y en tiempo real, entrenamiento con knowledge distillation, y despliegue con Docker.

## 📁 Estructura del Repositorio

```
.
├── app/                    # 🌐 Aplicación web (FastAPI + HTML/CSS/JS)
│   ├── backend/            #    API servidor (FastAPI + Whisper)
│   ├── frontend/           #    Interfaz web
│   ├── Dockerfile          #    Imagen slim (modelo montado como volumen)
│   └── Dockerfile.bundled  #    Imagen bundled (modelo incluido)
│
├── training/               # 🧠 Scripts de entrenamiento
│   ├── configs/            #    Scripts SLURM para HPC
│   ├── data/               #    Preparación de datasets
│   ├── train*.py           #    Scripts de fine-tuning
│   └── test.py             #    Evaluación del modelo
│
├── src/                    # 🏗️ Arquitectura de modelos
│   ├── teacher_model/      #    Modelo teacher
│   └── teacher_student_architecture/  # Knowledge distillation
│
├── scripts/                # 🔧 Utilidades
│   ├── convert_h5_to_ggml.py
│   ├── exportar_hf.py
│   ├── exportar_hf_bin.py
│   └── checking_cktp.py
│
├── demo/                   # 🎮 Demos Gradio
├── docs/                   # 📄 Informe final (LaTeX)
├── models/                 # 📦 Modelos y checkpoints (gitignored)
│
├── docker-compose.yml      # Configuración Docker
├── pyproject.toml           # Dependencias Python (uv/pip)
└── LICENSE
```

## 🚀 Inicio Rápido

### Requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (gestor de paquetes)
- FFmpeg
- Docker (opcional, para despliegue)

### Instalación Local

```bash
# Clonar el repositorio
git clone <repo-url> && cd End_2_End

# Instalar dependencias
uv sync

# Colocar el modelo fine-tuned en models/base_hf/
```

### Ejecutar la Aplicación Web

```bash
cd app
uvicorn backend.server:app --host 0.0.0.0 --port 7860
```

Abre `http://localhost:7860` en el navegador.

### Ejecutar con Docker

```bash
# CPU — modelo montado desde el host
docker compose up whisper-cpu-slim

# CPU — modelo incluido en la imagen
docker compose up whisper-cpu-bundled

# GPU (requiere nvidia-container-toolkit)
docker compose up whisper-gpu-slim
```

| Servicio | Imagen | Descripción |
|---|---|---|
| `whisper-cpu-slim` | ~1 GB | Modelo montado desde el host. CPU. |
| `whisper-gpu-slim` | ~1 GB | Modelo montado desde el host. GPU NVIDIA. |
| `whisper-cpu-bundled` | ~2 GB+ | Modelo incluido en la imagen. CPU. |
| `whisper-gpu-bundled` | ~2 GB+ | Modelo incluido en la imagen. GPU NVIDIA. |

## 🧠 Entrenamiento

Los scripts de entrenamiento están en `training/`. Para entrenamiento en un clúster HPC con SLURM:

```bash
sbatch training/configs/slurm_train.sh
```

Para preparar los datos:
```bash
python training/data/preparar_datos_completo.py
```

## 📄 Licencia

MIT — ver [LICENSE](LICENSE).
