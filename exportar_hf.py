import argparse
import os
import torch
from src.teacher_student_architecture.teacher_model import TeacherModel

def main():
    parser = argparse.ArgumentParser(description="Convierte un checkpoint .ckpt de Lightning a formato HuggingFace")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/last-v1.ckpt", help="Ruta al archivo .ckpt")
    parser.add_argument("--output_dir", type=str, default="modelo_exportado_hf", help="Carpeta donde se guardará el modelo")
    parser.add_argument("--base_model", type=str, default="openai/whisper-large-v3", help="Modelo base original (para instanciar antes de cargar pesos)")
    args = parser.parse_args()

    print(f"Cargando checkpoint desde {args.ckpt_path}...")
    
    # Cargamos el modelo TeacherModel sobrescribiendo el model_path (por si entrenaste en el cluster BSC_Marenostrum)
    try:
        model_lightning = TeacherModel.load_from_checkpoint(
            checkpoint_path=args.ckpt_path,
            model_path=args.base_model,
            strict=False,     # Permite cargar aunque haya llaves ligeremente distintas
            map_location=torch.device('cpu') # Para usarlo en tu PC local (CPU)
        )
    except Exception as e:
        print(f"Error al cargar el TeacherModel: {e}")
        print("\nAlternativa manual de extracción (si falla load_from_checkpoint):")
        
        # Extracción manual de pesos:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
        processor = WhisperProcessor.from_pretrained(args.base_model)
        
        # Filtrar solo las llaves que pertenecen a 'model.' en el state_dict de Lightning
        state_dict_hf = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        model.load_state_dict(state_dict_hf)
        model_lightning = type("Dummy", (), {"model": model, "processor": processor})()

    print("Checkpoint cargado correctamente.")
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Exportando el modelo HuggingFace a formato estándar en '{args.output_dir}'...")
    
    # Guardar modelo y procesador
    model_lightning.model.save_pretrained(args.output_dir)
    model_lightning.processor.save_pretrained(args.output_dir)
    
    print("\n✅ ¡Conversión completada con éxito!")
    print(f"Ahora puedes cargar este modelo en Gradio usando: model='{args.output_dir}'")

if __name__ == "__main__":
    main()
