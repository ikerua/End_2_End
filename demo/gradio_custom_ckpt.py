import gradio as gr
import numpy as np
import os
from transformers import pipeline
import torch

print("="*60)
print("🚀 CARGANDO MODELO CUSTOM DESDE CHECKPOINT...")
print("="*60)

# Ruta dinamica para evitar problemas si se corre en root o demo/
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_exportado_hf")

if not os.path.exists(MODEL_PATH):
    print(f"❌ ADVERTENCIA: No se encuentra la ruta {MODEL_PATH}")
    print("Por favor, ejecuta antes 'python exportar_hf.py' para generar la carpeta con el modelo HuggingFace.")
    print("Cargando modelo por defecto (openai/whisper-large-v3) mientras tanto...")
    MODEL_PATH = "openai/whisper-large-v3"

# Instanciar el pipeline indicando que el modelo es la ruta local exportada
transcriptor = pipeline(
    "automatic-speech-recognition",
    model=MODEL_PATH,      # Ruta de tu checkpoint exportado (contiene config.json, model.safetensors, etc)
    tokenizer=MODEL_PATH,  # Si el tokenizer está en la misma carpeta exportada
    feature_extractor=MODEL_PATH, # Si el feature_extractor está en la misma carpeta
    device="cuda"
)
print("✅ Modelo cargado!\n")

historial = []

def transcribir(audio, progress=gr.Progress()):
    """Transcribe audio con barra de progreso utilizando el modelo custom"""
    
    if audio is None:
        return "⚠️ No hay audio"
    
    try:
        progress(0, desc="📊 Analizando audio...")
        
        sample_rate, data = audio
        
        # Convertir
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        
        nivel = np.abs(data).mean()
        
        if nivel < 0.001:
            return "⚠️ Audio muy bajo"
        
        progress(0.5, desc="🎙️ Transcribiendo con Checkpoint Entrenado...")
        
        resultado = transcriptor(
            {"sampling_rate": sample_rate, "raw": data},
            return_timestamps=True,
            generate_kwargs={"language": "basque", "task": "transcribe"}
        )
        
        texto = resultado.get("text", "").strip()
        
        progress(1.0, desc="✅ Completado!")
        
        print(f"✅ Transcrito: '{texto}'")
        
        if texto and len(texto) > 2:
            historial.append(f"🗣️ {texto}")
            return "\n\n".join(historial)
        else:
            return "⚠️ No se detectó habla"
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return f"❌ Error: {str(e)}"

def limpiar():
    global historial
    historial = []
    return ""

# Interfaz
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🎤 Transcriptor de Voz - MODELO ENTRENADO")
    gr.Markdown("✅ **Funcionando con Checkpoint propio** (Basque)")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="🎙️ Grabadora o Sube tu Audio"
            )
            
            with gr.Row():
                btn_transcribir = gr.Button("🚀 Transcribir", variant="primary")
                btn_limpiar = gr.Button("🗑️ Limpiar")
            
            gr.Markdown("""
            ### 📝 Instrucciones:
            1. Sube tu archivo o graba con 🔴
            2. Habla claro (si grabas) y presiona ⏹️
            3. Haz click en el botón **🚀 Transcribir**
            4. **Espera la transcripción**
            """)
        
        with gr.Column(scale=2):
            output = gr.Textbox(
                label="📝 Transcripciones Generadas",
                lines=15,
                placeholder="Las transcripciones de tu modelo entrenado aparecerán aquí..."
            )
    
    # Eventos
    btn_transcribir.click(
        fn=transcribir,
        inputs=audio_input,
        outputs=output
    )
    
    btn_limpiar.click(fn=limpiar, outputs=output)

if __name__ == "__main__":
    print("🎉 Interfaz lista con modelo entrenado!")
    demo.queue().launch(
        max_threads=10,
        show_error=True
    )
