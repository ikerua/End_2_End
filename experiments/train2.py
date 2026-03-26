"""
train.py - Monolithic Training Script for Whisper Large V3 (Basque)
DEBUG EDITION: Inundado de prints para cazar el "Loss 1.570 congelado"
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
# DATA COLLATOR (Aadimos prints basicos aqui)
# ==========================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = []
        valid_sentences = []
        
        # Vamos a ver si el DataCollator esta haciendo algo
        num_vacios = 0

        for feature in features:
            audio_data = feature["audio"]
            sentence = feature.get("sentence", "")
            
            try:
                if "bytes" in audio_data and audio_data["bytes"] is not None:
                    audio_array, sampling_rate = sf.read(io.BytesIO(audio_data["bytes"]))
                else:
                    audio_array, sampling_rate = sf.read(audio_data["path"])
                
                if len(audio_array.shape) > 1:
                    audio_array = librosa.to_mono(audio_array.T)

                if len(audio_array) == 0:
                    raise ValueError("Audio vacio")

                if np.isnan(audio_array).any():
                    raise ValueError("NaN en array original")

                if sampling_rate != 16000:
                    audio_array = librosa.resample(y=audio_array.T, orig_sr=sampling_rate, target_sr=16000).T
                    sampling_rate = 16000

                extracted = self.processor.feature_extractor(
                    audio_array, sampling_rate=sampling_rate
                ).input_features[0]
                
                if np.isnan(extracted).any():
                    raise ValueError("NaN tras extraccion")

            except Exception as e:
                # No imprimimos cada audio para no saturar, pero los contamos
                num_vacios += 1
                extracted = np.zeros((80, 3000), dtype=np.float32) 
                sentence = "" 

            input_features.append({"input_features": extracted})
            valid_sentences.append(sentence)

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

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
        batch["num_errores"] = num_vacios # Pasamos el chivato a la GPU
        
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
        self.processor = None
        self.dataset = None

    def setup(self, stage=None):
        self.processor = WhisperProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.processor.tokenizer.set_prefix_tokens(language="basque", task="transcribe")
        self.dataset = load_from_disk(self.data_dir).with_format("torch").cast_column("audio", Audio(decode=False))
        print(" [SETUP] Dataset cargado en memoria correctamente.")

    def train_dataloader(self):
        collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        return DataLoader(self.dataset["train"], batch_size=self.batch_size, 
                          shuffle=True, collate_fn=collator, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        return DataLoader(self.dataset["validation"], batch_size=self.batch_size, 
                          shuffle=False, collate_fn=collator, num_workers=self.num_workers, pin_memory=True)

# ==========================================
# LIGHTNING MODEL MODULE (LLENO DE PRINTS)
# ==========================================
class WhisperLightning(pl.LightningModule):
    def __init__(self, model_path, learning_rate, weight_decay, warmup_steps, max_steps):
        super().__init__()
        self.save_hyperparameters()
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
        self.processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
        
        self.model.generation_config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="basque", task="transcribe")
        self.model.generation_config.suppress_tokens = []
        
        # Descongelado por si acaso
        # self.model.model.encoder.requires_grad_(False)

    def forward(self, input_features, labels):
        return self.model(input_features=input_features, labels=labels)

    def training_step(self, batch, batch_idx):
        # SOLO LA GPU MAESTRA IMPRIME PARA NO VOLVERNOS LOCOS
        is_master = self.global_rank == 0
        
        input_features = batch["input_features"]
        labels = batch["labels"]
        errores_collator = batch.get("num_errores", 0)

        # 1. Comprobacion de Tensores Puros
        if is_master and batch_idx % 20 == 0:
            print(f"\n--- DEBUG STEP {batch_idx} ---")
            print(f" [GPU 0] Errores lectura DataCollator: {errores_collator}")
            print(f" [GPU 0] Input Features Max: {input_features.max().item():.4f}, Min: {input_features.min().item():.4f}")
            print(f" [GPU 0] Numero total de etiquetas en el batch: {labels.numel()}")
            print(f" [GPU 0] Etiquetas a -100 (vacias): {(labels == -100).sum().item()}")

        # 2. Escudo contra Division por cero total
        if (labels == -100).all():
            if is_master: print(f" [ALERTA GPU 0] Batch {batch_idx} tiene TODAS las labels a -100. Retornando 0.0")
            return sum([p.sum() for p in self.parameters() if p.requires_grad]) * 0.0

        # 3. Forward real
        outputs = self(input_features=input_features, labels=labels)
        loss = outputs.loss

        # 4. Impresion del Loss Crudo (Antes del promedio de Lightning)
        if is_master and batch_idx % 20 == 0:
            print(f" [GPU 0] Loss crudo de la red: {loss.item():.5f}")

        # 5. Escudo contra NaN/Infinito
        if torch.isnan(loss) or torch.isinf(loss):
            if is_master: print(f" [FATAL GPU 0] El Loss ha dado NaN/Infinito en el batch {batch_idx}.")
            return sum([p.sum() for p in self.parameters() if p.requires_grad]) * 0.0

        # 6. Log normal
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_features = batch["input_features"]
        labels = batch["labels"]

        outputs = self(input_features=input_features, labels=labels)
        val_loss = outputs.loss
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        generated_ids = self.model.generate(
            input_features=input_features,
            max_new_tokens=225,
            language="basque",
            task="transcribe"
        )

        decoded_preds = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        labels = labels.clone()
        labels[labels == -100] = self.processor.tokenizer.pad_token_id
        decoded_labels = self.processor.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p if p.strip() else "EMPTY" for p in decoded_preds]
        decoded_labels = [l if l.strip() else "EMPTY" for l in decoded_labels]

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
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1) # CAMBIADO A 1 POR DEFECTO AHORA
    parser.add_argument("--gpus_per_node", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    
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
        precision="bf16-mixed", 
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_best, early_stopping, lr_monitor],
        logger=csv_logger,
        val_check_interval=0.25, 
        log_every_n_steps=10,
    )

    print("================================================================")
    print(" INICIANDO ENTRENAMIENTO DEBUG - WHISPER EUSKERA")
    print("================================================================")

    trainer.fit(model, datamodule=datamodule)

    print("Guardando el modelo final...")
    final_path = os.path.join(args.output_dir, "whisper-euskera-final")
    model.model.save_pretrained(final_path)
    model.processor.save_pretrained(final_path)
    print("Proceso terminado.")

if __name__ == "__main__":
    main()