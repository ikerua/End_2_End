import os
import random
import pandas as pd
from datasets import Dataset, DatasetDict, Audio

# Fijamos semilla para repetibilidad
random.seed(42)

# 1. Rutas base
BASE_DIR = "/gpfs/projects/ehpc485/tesi681824/transcriptor/datos"
OUT_DIR = "/gpfs/projects/ehpc485/tesi681824/transcriptor/dataset_unificado"

splits = {
    "train": {"audio": [], "sentence": []},
    "validation": {"audio": [], "sentence": []},
    "test": {"audio": [], "sentence": []}
}

print("1. Procesando Common Voice (Euskera)...")
cv_base = os.path.join(BASE_DIR, "cv-corpus-24.0-2025-12-05", "eu")
cv_clips = os.path.join(cv_base, "clips")

for split_name, hf_split in [("train", "train"), ("test", "test"), ("dev", "validation")]:
    tsv_path = os.path.join(cv_base, f"{split_name}.tsv")
    if os.path.exists(tsv_path):
        df = pd.read_csv(tsv_path, sep='\t')
        for _, row in df.iterrows():
            splits[hf_split]["audio"].append(os.path.join(cv_clips, row["path"]))
            splits[hf_split]["sentence"].append(row["sentence"].strip())

print("2. Procesando Parlamento (95% Train / 5% Validacion)...")
parl_audio = os.path.join(BASE_DIR, "train_audios")
tsv_parl_train = os.path.join(BASE_DIR, "train.tsv")

if os.path.exists(tsv_parl_train):
    df_parl = pd.read_csv(tsv_parl_train, sep='\t')
    parl_items = []
    
    for _, row in df_parl.iterrows():
        parl_items.append({
            "audio": os.path.join(parl_audio, row["path"]),
            "sentence": row["sentence"].strip()
        })
    
    random.shuffle(parl_items)
    corte_parl = int(len(parl_items) * 0.95)
    
    for item in parl_items[:corte_parl]:
        splits["train"]["audio"].append(item["audio"])
        splits["train"]["sentence"].append(item["sentence"])
    for item in parl_items[corte_parl:]:
        splits["validation"]["audio"].append(item["audio"])
        splits["validation"]["sentence"].append(item["sentence"])

print("3. Procesando eu_es Male/Female (95% Train / 5% Validacion)...")
eues_items = []
for genero in ["male", "female"]:
    tsv_path = os.path.join(BASE_DIR, f"line_index_{genero}.tsv")
    audio_dir = os.path.join(BASE_DIR, f"eu_es_{genero}")
    if os.path.exists(tsv_path):
        df_gen = pd.read_csv(tsv_path, sep='\t', header=None, names=["id", "texto"])
        for _, row in df_gen.iterrows():
            ruta_audio = os.path.join(audio_dir, f"{row['id']}.wav")
            eues_items.append({
                "audio": ruta_audio,
                "sentence": row["texto"].strip()
            })

random.shuffle(eues_items)
corte_eues = int(len(eues_items) * 0.95)

for item in eues_items[:corte_eues]:
    splits["train"]["audio"].append(item["audio"])
    splits["train"]["sentence"].append(item["sentence"])
for item in eues_items[corte_eues:]:
    splits["validation"]["audio"].append(item["audio"])
    splits["validation"]["sentence"].append(item["sentence"])

print("\n--- RESUMEN DE DATOS RECOPILADOS ---")
for split, data in splits.items():
    print(f"{split.upper()}: {len(data['audio'])} audios")

print("\n4. Generando el Dataset y forzando resampling a 16kHz...")
ds_dict = {}
for split, data in splits.items():
    if len(data["audio"]) > 0:
        ds = Dataset.from_dict(data)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        ds_dict[split] = ds

dataset_completo = DatasetDict(ds_dict)

print("\nGuardando a disco...")
dataset_completo.save_to_disk(OUT_DIR)
print(f"Dataset unificado guardado con exito en {OUT_DIR}")