"""
train.py — Main training script for Whisper Large V3 fine-tuning on Basque speech.

Designed for BSC MareNostrum:
  - 2 nodes × 4 GPUs each = 8 GPUs total
  - Full FP32 precision (no hardware limitations)
  - DDP strategy (DistributedDataParallel)

Usage (local test):
    python train.py --model_path ./whisper-large-v3 --data_dir ./data

Usage (MareNostrum via SLURM):
    sbatch slurm_train.sh
"""

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from lib.teacher_model.model import WhisperFinetuner
from lib.teacher_model.dataset import EuskeraTranscriptorDataModule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper Large V3 for Basque transcription"
    )

    # Paths
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the Whisper Large V3 model directory (with .msgpack weights)",
    )
    parser.add_argument(
        "--data_dir_clean",
        type=str,
        default="./data/clean",
        help="Path to clean audio data",
    )
    parser.add_argument(
        "--data_dir_noisy",
        type=str,
        default="./data/noisy",
        help="Path to noisy audio data",
    )
    parser.add_argument(
        "--use_clean_data",
        action="store_true",
        default=True,
        help="Whether to use clean data (default: True)",
    )

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch_size × num_gpus × this)",
    )

    # Infrastructure
    parser.add_argument(
        "--num_nodes", type=int, default=2, help="Number of compute nodes"
    )
    parser.add_argument(
        "--gpus_per_node",
        type=int,
        default=4,
        help="Number of GPUs per node",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory for saving model checkpoints",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory for TensorBoard logs",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ─── 1. Initialize the model ──────────────────────────────────────
    model = WhisperFinetuner(
        model_name_or_path=args.model_path,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )

    # ─── 2. Initialize the data module ────────────────────────────────
    datamodule = EuskeraTranscriptorDataModule(
        data_dir=args.data_dir_clean if args.use_clean_data else args.data_dir_noisy,
        model_name_or_path=args.model_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ─── 3. Callbacks ─────────────────────────────────────────────────

    # Save best model based on validation WER
    checkpoint_best = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="whisper-euskera-best-{epoch:02d}-{val_wer:.4f}",
        monitor="val_wer",
        mode="min",       # Lower WER = better
        save_top_k=3,     # Keep top 3 models
        verbose=True,
    )

    # Save latest model every epoch
    checkpoint_last = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="whisper-euskera-last-{epoch:02d}",
        save_last=True,
        every_n_epochs=1,
    )

    # Stop training if WER doesn't improve for 5 epochs
    early_stopping = EarlyStopping(
        monitor="val_wer",
        mode="min",
        patience=5,
        verbose=True,
    )

    # Log learning rate to TensorBoard
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ─── 4. Logger ────────────────────────────────────────────────────
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="whisper_euskera_teacher",
    )

    # ─── 5. Trainer — MareNostrum configuration ──────────────────────
    #
    #   2 nodes × 4 GPUs = 8 GPUs total
    #   Full FP32 precision (precision="32-true")
    #   DDP strategy for multi-node communication via NCCL
    #
    #   Effective batch size = batch_size × 8 GPUs × accumulate_grad_batches
    #   Example: 16 × 8 × 1 = 128 samples per optimizer step
    #
    trainer = pl.Trainer(
        # Multi-node, multi-GPU
        num_nodes=args.num_nodes,
        devices=args.gpus_per_node,
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator="gpu",

        # Full precision — no FP16/BF16 limitation
        precision="32-true",

        # Training duration
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,

        # Gradient accumulation
        accumulate_grad_batches=args.accumulate_grad_batches,

        # Gradient clipping to prevent exploding gradients
        gradient_clip_val=1.0,

        # Callbacks and logging
        callbacks=[checkpoint_best, checkpoint_last, early_stopping, lr_monitor],
        logger=logger,

        # Validation frequency
        val_check_interval=0.25,  # Validate 4 times per epoch

        # Logging frequency
        log_every_n_steps=10,

        # Deterministic for reproducibility
        deterministic=False,  # True can slow down training
    )

    # ─── 6. Train! ────────────────────────────────────────────────────
    print("=" * 70)
    print("  WHISPER LARGE V3 — INTENSIVE BASQUE FINE-TUNING")
    print(f"  Nodes: {args.num_nodes} | GPUs/node: {args.gpus_per_node}")
    print(f"  Effective batch size: {args.batch_size} × {args.num_nodes * args.gpus_per_node} × {args.accumulate_grad_batches} = {args.batch_size * args.num_nodes * args.gpus_per_node * args.accumulate_grad_batches}")
    print(f"  Precision: FP32 (full)")
    print(f"  Learning rate: {args.learning_rate} (warmup: {args.warmup_steps} steps)")
    print(f"  Max epochs: {args.max_epochs} | Max steps: {args.max_steps}")
    print("=" * 70)

    trainer.fit(model, datamodule=datamodule)

    # ─── 7. Save the final HuggingFace model for later use ────────────
    print("\nSaving final model in HuggingFace format...")
    model.model.save_pretrained(f"{args.output_dir}/whisper-euskera-final")
    model.processor.save_pretrained(f"{args.output_dir}/whisper-euskera-final")
    print(f"Model saved to {args.output_dir}/whisper-euskera-final")


if __name__ == "__main__":
    main()
