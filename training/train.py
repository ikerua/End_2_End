"""
train.py - Monolithic Training Script for Whisper Large V3 (Basque)
Adapted for BSC MareNostrum 5 (Offline Mode, Scratch storage, Multi-Node DDP).
No external local libraries used to prevent import/path errors.
"""

import os
import argparse
import torch
import jiwer
import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import io
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_from_disk, Audio
from torch.utils.data import DataLoader
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy

# 1. BLOQUEO OFFLINE (VITAL PARA EL BSC)
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"

# ==========================================
# DATA COLLATOR
# ==========================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = []
        valid_sentences = []

        for feature in features:
            audio_data = feature["audio"]
            sentence = feature.get("sentence", "")
            
            try:
                # 1. Leer audio
                if "bytes" in audio_data and audio_data["bytes"] is not None:
                    audio_array, sampling_rate = sf.read(io.BytesIO(audio_data["bytes"]))
                else:
                    audio_array, sampling_rate = sf.read(audio_data["path"])
                # 1. FORZAR MONO (CRITICO: Whisper colapsa con audios estereo)
                if len(audio_array.shape) > 1:
                    audio_array = librosa.to_mono(audio_array.T)

                # 2. Escudo contra audios vacios (0 segundos)
                if len(audio_array) == 0:
                    raise ValueError("Audio de longitud 0 detectado")

                # 3. Escudo contra ruido infinito o NaNs de origen
                if np.isnan(audio_array).any():
                    raise ValueError("Valores NaN detectados en el array original")

                # 4. Re-muestreo a 16kHz
                if sampling_rate != 16000:
                    audio_array = librosa.resample(y=audio_array.T, orig_sr=sampling_rate, target_sr=16000).T
                    sampling_rate = 16000

                # 5. Extraccion de caracteristicas
                extracted = self.processor.feature_extractor(
                    audio_array, sampling_rate=sampling_rate
                ).input_features[0]
                
                # 6. Escudo final post-extraccion
                if np.isnan(extracted).any():
                    raise ValueError("Valores NaN tras extraer espectrograma")

            except Exception as e:
                # Si CUALQUIER COSA falla, ponemos un segundo de silencio puro y vaciamos la frase
                print(f" [WARNING] Audio descartado por error: {e}")
                extracted = np.zeros((80, 3000), dtype=np.float32) # Espectrograma vacio estandar
                sentence = "" # No hay frase que aprender de un silencio

            input_features.append({"input_features": extracted})
            valid_sentences.append(sentence)

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Tokenizamos las frases validas
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
        return batch
# ==========================================
# DATAMODULE (PYTORCH LIGHTNING)
# ==========================================
class WhisperDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, model_path, batch_size=8, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.model_path = model_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.processor = None
        self.dataset = None

    def setup(self, stage=None):
        self.processor = WhisperProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.processor.tokenizer.set_prefix_tokens(language="basque", task="transcribe")
        self.dataset = load_from_disk(self.data_dir).with_format("torch").cast_column("audio", Audio(decode=False))

    def train_dataloader(self):
        collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        return DataLoader(self.dataset["train"], batch_size=self.batch_size, 
                          shuffle=True, collate_fn=collator, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        return DataLoader(self.dataset["validation"], batch_size=self.batch_size, 
                          shuffle=False, collate_fn=collator, num_workers=self.num_workers, pin_memory=True)

# ==========================================
# LIGHTNING MODEL MODULE
# ==========================================
class WhisperLightning(pl.LightningModule):
    def __init__(self, model_path, learning_rate, weight_decay, warmup_steps, max_steps):
        super().__init__()
        self.save_hyperparameters()
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
        
        # ESCUDO MATEMATICO Congelamos el encoder para evitar Gradient Explosion
        #self.model.model.encoder.requires_grad_(False)
        
        self.processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
        
        # Forzar Euskera
        self.model.generation_config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="basque", task="transcribe")
        self.model.generation_config.suppress_tokens = []
    def forward(self, input_features, labels):
        return self.model(input_features=input_features, labels=labels)

    def training_step(self, batch, batch_idx):
        # 1. Limpieza extrema de tensores de audio por si cuela algun infinito
        input_features = torch.nan_to_num(batch["input_features"], nan=0.0, posinf=0.0, neginf=0.0)
        labels = batch["labels"]

        # 2. ESCUDO 1: Si TODAS las etiquetas son -100 (texto vacio) da division por cero.
        if (labels == -100).all():
            # Dummy loss: Multiplicamos los pesos por 0 para que de exactamente 0.0
            # Esto permite hacer backward() y sincronizar las GPUs sin colgar el MareNostrum
            dummy_loss = sum([p.sum() for p in self.parameters() if p.requires_grad]) * 0.0
            return dummy_loss

        # 3. Forward pass normal
        outputs = self(input_features=input_features, labels=labels)
        loss = outputs.loss

        # 4. ESCUDO 2: Si la atencion matematica del Transformer explota y da NaN
        if torch.isnan(loss) or torch.isinf(loss):
            dummy_loss = sum([p.sum() for p in self.parameters() if p.requires_grad]) * 0.0
            return dummy_loss

        # Solo logueamos la perdida si es un numero real sano
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_features = batch["input_features"]
        labels = batch["labels"]

        # 1. Calculo de Loss
        outputs = self(input_features=input_features, labels=labels)
        val_loss = outputs.loss
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # 2. Generacion de texto para metricas
        generated_ids = self.model.generate(
            input_features=input_features,
            max_new_tokens=225,
            language="basque",
            task="transcribe"
        )

        # Decodificar predicciones
        decoded_preds = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Decodificar labels reales (reemplazando -100)
        labels = labels.clone()
        labels[labels == -100] = self.processor.tokenizer.pad_token_id
        decoded_labels = self.processor.batch_decode(labels, skip_special_tokens=True)

        # Limpiar strings vacios para jiwer
        decoded_preds = [p if p.strip() else "EMPTY" for p in decoded_preds]
        decoded_labels = [l if l.strip() else "EMPTY" for l in decoded_labels]

        # 3. Calculo de WER y CER
        wer = jiwer.wer(decoded_labels, decoded_preds)
        cer = jiwer.cer(decoded_labels, decoded_preds)

        self.log("val_wer", wer, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_cer", cer, on_epoch=True, prog_bar=True, sync_dist=True)

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.max_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}
        }

# ==========================================
# MAIN ROUTINE
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/gpfs/projects/ehpc485/tesi681824/transcriptor/modelo_whisper_large_v3")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset (will be passed as TMPDIR by SLURM)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=2)
    parser.add_argument("--gpus_per_node", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    
    # SCRATCH PATHS
    parser.add_argument("--output_dir", type=str, default="/gpfs/scratch/ehpc485/tesi681824/whisper_eu_out/checkpoints")
    parser.add_argument("--log_dir", type=str, default="/gpfs/scratch/ehpc485/tesi681824/whisper_eu_out/logs")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    model = WhisperLightning(
        model_path=args.model_path,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )

    datamodule = WhisperDataModule(
        data_dir=args.data_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    checkpoint_best = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="whisper-best-epoch={epoch:02d}-wer={val_wer:.4f}-cer={val_cer:.4f}",
        monitor="val_wer",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stopping = EarlyStopping(monitor="val_wer", mode="min", patience=5, verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    csv_logger = CSVLogger(save_dir=args.log_dir, name="csv_logs")

    trainer = pl.Trainer(
        num_nodes=args.num_nodes,
        devices=args.gpus_per_node,
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator="gpu",
        precision="bf16-mixed", # Super-optimizacion para H100
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_best, early_stopping, lr_monitor],
        logger=csv_logger,
        val_check_interval=0.25, # Valida 4 veces por epoca
        log_every_n_steps=10,
    )

    print("================================================================")
    print(" INICIANDO ENTRENAMIENTO WHISPER - EUSKERA (OFFLINE)")
    print(f" Nodos: {args.num_nodes} | GPUs por nodo: {args.gpus_per_node}")
    print(f" Precision: bfloat16 mixed (Optimizado para H100)")
    print(f" Checkpoints guardados en: {args.output_dir}")
    print("================================================================")

    trainer.fit(model, datamodule=datamodule)

    print("Guardando el modelo final...")
    final_path = os.path.join(args.output_dir, "whisper-euskera-final")
    model.model.save_pretrained(final_path)
    model.processor.save_pretrained(final_path)
    print("Proceso terminado correctamente.")

if __name__ == "__main__":
    main()