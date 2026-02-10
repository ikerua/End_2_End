import gradio as gr
import numpy as np
from transformers import pipeline
import torch

print("="*60)
print("🚀 CARGANDO MODELO...")
print("="*60)

transcriptor = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    device="cpu"
)
print("✅ Modelo cargado!\n")

historial = []

def transcribir(audio, progress=gr.Progress()):
    """Transcribe audio con barra de progreso"""
    
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
        
        progress(0.5, desc="🎙️ Transcribiendo...")
        
        # Transcribir
        resultado = transcriptor(
            {"sampling_rate": sample_rate, "raw": data},
            generate_kwargs={"language": "basque", "task": "transcribe"}
        )
        
        texto = resultado["text"].strip()
        
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

# Interfaz con configuración mejorada
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🎤 Transcriptor de Voz")
    gr.Markdown("✅ **Funcionando correctamente** - Whisper Base")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="🎙️ Grabadora"
            )
            
            btn_limpiar = gr.Button("🗑️ Limpiar")
            
            gr.Markdown("""
            ### 📝 Instrucciones:
            1. Presiona 🔴
            2. Habla claro
            3. Presiona ⏹️
            4. **Espera 5-10 segundos**
            
            ⏳ La transcripción tarda un poco, es normal
            """)
        
        with gr.Column(scale=2):
            output = gr.Textbox(
                label="📝 Transcripciones",
                lines=15,
                placeholder="Transcripciones aquí..."
            )
    
    # Eventos
    audio_input.change(
        fn=transcribir,
        inputs=audio_input,
        outputs=output
    )
    
    btn_limpiar.click(fn=limpiar, outputs=output)

print("🎉 Interfaz lista!")
print("⏳ Las transcripciones pueden tardar 5-10 segundos\n")

# Lanzar con timeouts más largos
demo.queue().launch(
    max_threads=10,
    show_error=True
)