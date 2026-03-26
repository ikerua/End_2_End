"""
evaluate_models.py - Comparativa de 5 modelos Whisper en el dataset Test
"""

import os
import torch
import jiwer
import numpy as np
import soundfile as sf
import io
import librosa
from datasets import load_from_disk, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm

# Bloqueo Offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def load_custom_checkpoint(base_model_path, ckpt_path, is_distilled=False):
    """
    Carga la arquitectura base y le inyecta los pesos del .ckpt de Lightning.
    Limpia los prefijos correctamente sin romper las llaves internas.
    """
    print(f"Cargando arquitectura desde: {base_model_path}")
    model = WhisperForConditionalGeneration.from_pretrained(base_model_path, local_files_only=True)
    
    print(f"Inyectando pesos desde: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        # FORMA CORRECTA Cortar solo la longitud exacta del prefijo
        if k.startswith("student_model."):
            new_key = k[len("student_model."):] 
            cleaned_state_dict[new_key] = v
        elif k.startswith("teacher_model."):
            continue
        elif k.startswith("model."):
            new_key = k[len("model."):] # Corta solo el primer model
            cleaned_state_dict[new_key] = v
        elif k.startswith("mel_projection"):
            continue
        else:
            cleaned_state_dict[k] = v
            
    # strict=False ignora lo que sobre, pero guarda lo que falta
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    
    print(f" -> Capas no encontradas (Missing): {len(missing)}")
    if len(missing) > 5:
        print(f" ALERTA Cuidado! Faltan demasiadas capas: {missing[:3]}...")
        
    return model
def evaluate_single_model(model_name, model, processor, dataset, device="cuda"):
    print(f"\n======================================")
    print(f" EVALUANDO MODELO: {model_name}")
    print(f"======================================")
    
    model = model.to(device)
    model.eval()
    
    # Forzar el idioma en la generacion
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="basque", task="transcribe")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for item in tqdm(dataset, desc=f"Procesando {model_name}"):
            audio_data = item["audio"]
            real_text = item.get("sentence", "").strip()
            
            if not real_text:
                continue
                
            # Leer audio robustamente
            if "bytes" in audio_data and audio_data["bytes"] is not None:
                audio_array, sr = sf.read(io.BytesIO(audio_data["bytes"]))
            else:
                audio_array, sr = sf.read(audio_data["path"])
                
            if len(audio_array.shape) > 1:
                audio_array = librosa.to_mono(audio_array.T)
                
            if sr != 16000:
                audio_array = librosa.resample(y=audio_array.T, orig_sr=sr, target_sr=16000).T
                
            # Extraer features
            inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device)
            
            # Asegurar tipo de datos (FP32 o FP16)
            input_features = input_features.to(model.model.encoder.conv1.weight.dtype)
            
            # Generar texto
            generated_ids = model.generate(
                input_features=input_features,
                max_new_tokens=225,
                forced_decoder_ids=forced_decoder_ids,
                suppress_tokens=[]
            )
            
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            all_preds.append(pred_text if pred_text else "EMPTY")
            all_labels.append(real_text)
            
    wer = jiwer.wer(all_labels, all_preds)
    cer = jiwer.cer(all_labels, all_preds)
    
    print(f" -> RESULTADOS {model_name}:")
    print(f" -> WER: {wer * 100:.2f}%")
    print(f" -> CER: {cer * 100:.2f}%")
    
    # Liberar memoria de la grafica
    model = model.to("cpu")
    torch.cuda.empty_cache()
    
    return wer, cer

def main():
    # 0. Definir rutas
    test_data_path = "/gpfs/projects/ehpc485/tesi681824/transcriptor/dataset_unificado/test"
    
    path_base_original = "/gpfs/projects/ehpc485/tesi681824/transcriptor/modelo_whisper_base"
    path_large_original = "/gpfs/projects/ehpc485/tesi681824/transcriptor/modelo_whisper_large_v3"
    
    ckpt_large_ft = "/gpfs/scratch/ehpc485/tesi681824/whisper_eu_out/checkpoints/last-v2.ckpt"
    ckpt_base_ft = "/gpfs/scratch/ehpc485/tesi681824/whisper_eu_out_base/checkpoints/last.ckpt"
    ckpt_student = "/gpfs/scratch/ehpc485/tesi681824/whisper_distill_out/checkpoints_fase2/last.ckpt"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")
    
    # 1. Cargar Dataset
    print(f"Cargando dataset de test desde: {test_data_path}")
    test_dataset = load_from_disk(test_data_path).cast_column("audio", Audio(decode=False))
    
    resultados = {}
    
    # --- MODELO 1: BASE ORIGINAL ---
    # processor_base = WhisperProcessor.from_pretrained(path_base_original, local_files_only=True)
    # model_1 = WhisperForConditionalGeneration.from_pretrained(path_base_original, local_files_only=True)
    # wer1, cer1 = evaluate_single_model("1. Whisper Base Original", model_1, processor_base, test_dataset, device)
    # resultados["Base Original"] = (wer1, cer1)
    # del model_1
    resultados["Base Original"] = (1.0394, 0.3224)
    
    # --- MODELO 2: LARGE ORIGINAL ---
    # processor_large = WhisperProcessor.from_pretrained(path_large_original, local_files_only=True)
    # model_2 = WhisperForConditionalGeneration.from_pretrained(path_large_original, local_files_only=True)
    # wer2, cer2 = evaluate_single_model("2. Whisper Large V3 Original", model_2, processor_large, test_dataset, device)
    # resultados["Large V3 Original"] = (wer2, cer2)
    # del model_2
    resultados["Large V3 Original"] = (0.4784, 0.0928)

    processor_base = WhisperProcessor.from_pretrained(path_base_original, local_files_only=True)
    processor_large = WhisperProcessor.from_pretrained(path_large_original, local_files_only=True)
    
    # --- MODELO 3: BASE FINE-TUNED ---
    model_3 = load_custom_checkpoint(path_base_original, ckpt_base_ft)
    wer3, cer3 = evaluate_single_model("3. Whisper Base Fine-Tuned", model_3, processor_base, test_dataset, device)
    resultados["Base Fine-Tuned"] = (wer3, cer3)
    del model_3
    
    # --- MODELO 4: LARGE FINE-TUNED ---
    model_4 = load_custom_checkpoint(path_large_original, ckpt_large_ft)
    wer4, cer4 = evaluate_single_model("4. Whisper Large V3 Fine-Tuned", model_4, processor_large, test_dataset, device)
    resultados["Large V3 Fine-Tuned"] = (wer4, cer4)
    del model_4
    
    # --- MODELO 5: TEACHER-STUDENT DISTILLED ---
    model_5 = load_custom_checkpoint(path_base_original, ckpt_student, is_distilled=True)
    wer5, cer5 = evaluate_single_model("5. Whisper Base Destilado (Teacher-Student)", model_5, processor_base, test_dataset, device)
    resultados["Base Destilado (T-S)"] = (wer5, cer5)
    del model_5
    
    # --- IMPRIMIR TABLA FINAL ---
    print("\n" + "="*50)
    print(" RESUMEN FINAL DE EVALUACION (TEST SET)")
    print("="*50)
    print(f"{'Modelo':<35} | {'WER (%)':<8} | {'CER (%)':<8}")
    print("-" * 55)
    for nombre, (wer, cer) in resultados.items():
        print(f"{nombre:<35} | {wer*100:>7.2f}% | {cer*100:>7.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()