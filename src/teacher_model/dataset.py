"""
This module contains the DataModule for the Euskera Transcriptor task.
It handles loading, splitting, scaling, and creating data loaders for
the dataset.
"""

__docformat__ = "numpy"

import os
import torch
import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import io
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_from_disk, Audio
from torch.utils.data import DataLoader

from transformers import WhisperProcessor
# ==========================================
# DATA COLLATOR (FIX 128 BANDS + DEBUG)
# ==========================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = []
        valid_sentences = []
        num_vacios = 0

        for idx, feature in enumerate(features):
            audio_data = feature["audio"]
            sentence = feature.get("sentence", "")
            
            try:
                # 1. Texto vacio
                if not sentence or len(sentence.strip()) == 0:
                    raise ValueError("Texto completamente vacio")

                # 2. Leer audio
                if "bytes" in audio_data and audio_data["bytes"] is not None:
                    audio_array, sampling_rate = sf.read(io.BytesIO(audio_data["bytes"]))
                else:
                    audio_array, sampling_rate = sf.read(audio_data["path"])
                
                # 3. Forzar Mono
                if len(audio_array.shape) > 1:
                    audio_array = librosa.to_mono(audio_array.T)

                # 4. Longitud minima
                if len(audio_array) < 1600:
                    raise ValueError(f"Audio demasiado corto ({len(audio_array)} muestras)")

                # 5. Valores corruptos
                if np.isnan(audio_array).any() or np.isinf(audio_array).any():
                    raise ValueError("NaN/Infinito detectado en el audio original")

                # 6. Remuestreo a 16KHz
                if sampling_rate != 16000:
                    audio_array = librosa.resample(y=audio_array.T, orig_sr=sampling_rate, target_sr=16000).T
                    sampling_rate = 16000

                # 7. Extraccion
                extracted = self.processor.feature_extractor(
                    audio_array, sampling_rate=sampling_rate
                ).input_features[0]
                
                if np.isnan(extracted).any() or np.isinf(extracted).any():
                    raise ValueError("NaN tras extraccion del espectrograma")

            except Exception as e:
                # DEBUG PRINT: Solo logueamos si hay error
                print(f" [DataCollator] ERROR en audio {idx}: {e}. Sustituyendo por dummy 128-Mel.")
                num_vacios += 1
                # FIX CRITICO 1: 128 MEL BANDS PARA WHISPER V3 (No 80)
                extracted = np.zeros((128, 3000), dtype=np.float32) 
                sentence = "Audio akatsa" 

            input_features.append({"input_features": extracted})
            valid_sentences.append(sentence)

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Tokenizacion con guillotina para evitar OutOfBounds
        labels_batch = self.processor.tokenizer(
            valid_sentences, 
            padding="longest",
            truncation=True, 
            max_length=448,
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        batch["num_errores"] = num_vacios 
        return batch

# ==========================================
# DATAMODULE 
# ==========================================
class WhisperDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, model_path, batch_size=8, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.model_path = model_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.processor = WhisperProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.processor.tokenizer.set_prefix_tokens(language="basque", task="transcribe")
        self.dataset = load_from_disk(self.data_dir).with_format("torch").cast_column("audio", Audio(decode=False))
        print(" [SETUP] Dataset cargado en memoria correctamente.")

    def train_dataloader(self):
        collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        return DataLoader(self.dataset["train"], batch_size=self.batch_size, shuffle=True, collate_fn=collator, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        return DataLoader(self.dataset["validation"], batch_size=self.batch_size, shuffle=False, collate_fn=collator, num_workers=self.num_workers, pin_memory=True)
