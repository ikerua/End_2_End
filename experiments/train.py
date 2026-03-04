"""
train.py - Monolithic Training Script for Whisper Large V3 (Basque)
DEBUG EXTREMO + FIX 128 MEL BANDS + USE_CACHE FALSE + ENCODER FROZEN
"""

import os
import argparse
import torch

import pytorch_lightning as pl
import soundfile as sf
import io
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy

from src.teacher_model.model import WhisperLightning
from src.teacher_model.data import WhisperDataModule

# 1. BLOQUEO OFFLINE (VITAL PARA EL BSC)
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"

# ==========================================
# MAIN ROUTINE
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/gpfs/projects/ehpc485/tesi681824/transcriptor/modelo_whisper_large_v3")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--gpus_per_node", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    
    parser.add_argument("--output_dir", type=str, default="/gpfs/scratch/ehpc485/tesi681824/whisper_eu_out/checkpoints")
    parser.add_argument("--log_dir", type=str, default="/gpfs/scratch/ehpc485/tesi681824/whisper_eu_out/logs")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    model = WhisperLightning(
        model_path=args.model_path, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, max_steps=args.max_steps
    )
    datamodule = WhisperDataModule(data_dir=args.data_dir, model_path=args.model_path, batch_size=args.batch_size, num_workers=args.num_workers)

    checkpoint_best = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="whisper-best-epoch={epoch:02d}-wer={val_wer:.4f}",
        monitor="val_wer",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stopping = EarlyStopping(monitor="val_wer", mode="min", patience=5, verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    csv_logger = CSVLogger(save_dir=args.log_dir, name="csv_logs")

    trainer = pl.Trainer(
        num_nodes=args.num_nodes, devices=args.gpus_per_node, strategy=DDPStrategy(find_unused_parameters=False),
        accelerator="gpu", precision="32-true", max_epochs=args.max_epochs, max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches, gradient_clip_val=1.0,
        callbacks=[checkpoint_best, early_stopping, lr_monitor], logger=csv_logger,
        val_check_interval=0.25, log_every_n_steps=10,
    )

    print("================================================================")
    print(" INICIANDO ENTRENAMIENTO BLINDADO - WHISPER EUSKERA V3")
    print("================================================================")

    trainer.fit(model, datamodule=datamodule)

    print("Guardando el modelo final...")
    final_path = os.path.join(args.output_dir, "whisper-euskera-final")
    model.model.save_pretrained(final_path)
    model.processor.save_pretrained(final_path)
    print("Proceso terminado.")

if __name__ == "__main__":
    main()