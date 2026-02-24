"""
train_kd.py — Knowledge Distillation training script.

Trains a smaller Student Whisper model using the fine-tuned Teacher (Whisper Large V3)
to transfer its Basque transcription capabilities.

Designed for BSC MareNostrum:
  - 2 nodes × 4 GPUs each = 8 GPUs total
  - Full FP32 precision
  - DDP strategy

Usage (local test):
    python -m experiments.train_kd \\
        --teacher_path ./checkpoints/whisper-euskera-final \\
        --student_path openai/whisper-small \\
        --data_dir_clean ./data/clean

Usage (MareNostrum via SLURM):
    sbatch experiments/slurm_train_kd.sh
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

from lib.teacher_student_architecture.model import WhisperKnowledgeDistillation
from lib.teacher_student_architecture.dataset import EuskeraTranscriptorDataModule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation: Whisper Large V3 → Whisper Small/Tiny"
    )

    # Model paths
    parser.add_argument(
        "--teacher_path",
        type=str,
        required=True,
        help="Path to the fine-tuned Teacher model (Whisper Large V3)",
    )
    parser.add_argument(
        "--student_path",
        type=str,
        default="openai/whisper-small",
        help="Path or HuggingFace ID for the Student model (e.g. openai/whisper-small, openai/whisper-tiny)",
    )
    parser.add_argument(
        "--processor_path",
        type=str,
        default=None,
        help="Path to the WhisperProcessor (defaults to student_path)",
    )

    # Data paths
    parser.add_argument("--data_dir_clean", type=str, default="./data/clean")
    parser.add_argument("--data_dir_noisy", type=str, default="./data/noisy")
    parser.add_argument("--use_clean_data", action="store_true", default=True)

    # KD Hyperparameters
    parser.add_argument(
        "--temperature", type=float, default=2.0,
        help="Temperature for softening logits (higher = softer). Default: 2.0",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.7,
        help="Weight for soft loss. Total = α*soft + (1-α)*hard. Default: 0.7",
    )

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=80000)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    # Infrastructure
    parser.add_argument("--num_nodes", type=int, default=2)
    parser.add_argument("--gpus_per_node", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/kd")
    parser.add_argument("--log_dir", type=str, default="./logs")

    return parser.parse_args()


def main():
    args = parse_args()

    # ─── 1. Initialize the KD model ──────────────────────────────────
    model = WhisperKnowledgeDistillation(
        teacher_model_path=args.teacher_path,
        student_model_path=args.student_path,
        processor_path=args.processor_path,
        temperature=args.temperature,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )

    # ─── 2. Initialize the data module ────────────────────────────────
    # Use the Student's processor path for tokenization
    data_model_path = args.processor_path or args.student_path
    datamodule = EuskeraTranscriptorDataModule(
        data_dir=args.data_dir_clean if args.use_clean_data else args.data_dir_noisy,
        model_name_or_path=data_model_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ─── 3. Callbacks ─────────────────────────────────────────────────

    # Save best Student model based on validation WER
    checkpoint_best = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="student-best-{epoch:02d}-{val_wer:.4f}",
        monitor="val_wer",
        mode="min",
        save_top_k=3,
        verbose=True,
    )

    checkpoint_last = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="student-last-{epoch:02d}",
        save_last=True,
        every_n_epochs=1,
    )

    early_stopping = EarlyStopping(
        monitor="val_wer",
        mode="min",
        patience=7,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ─── 4. Logger ────────────────────────────────────────────────────
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="whisper_kd_student",
    )

    # ─── 5. Trainer ───────────────────────────────────────────────────
    eff_batch = args.batch_size * args.num_nodes * args.gpus_per_node * args.accumulate_grad_batches

    trainer = pl.Trainer(
        num_nodes=args.num_nodes,
        devices=args.gpus_per_node,
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator="gpu",
        precision="32-true",
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_best, checkpoint_last, early_stopping, lr_monitor],
        logger=logger,
        val_check_interval=0.25,
        log_every_n_steps=10,
        deterministic=False,
    )

    # ─── 6. Train! ────────────────────────────────────────────────────
    print("=" * 70)
    print("  KNOWLEDGE DISTILLATION: Teacher → Student")
    print(f"  Teacher: {args.teacher_path}")
    print(f"  Student: {args.student_path}")
    print(f"  Temperature: {args.temperature} | Alpha: {args.alpha}")
    print(f"  Nodes: {args.num_nodes} | GPUs/node: {args.gpus_per_node}")
    print(f"  Effective batch size: {eff_batch}")
    print(f"  Precision: FP32 (full)")
    print(f"  Learning rate: {args.learning_rate} (warmup: {args.warmup_steps} steps)")
    print(f"  Max epochs: {args.max_epochs} | Max steps: {args.max_steps}")
    print("=" * 70)

    trainer.fit(model, datamodule=datamodule)

    # ─── 7. Save the final Student in HuggingFace format ──────────────
    print("\nSaving final Student model in HuggingFace format...")
    model.student.save_pretrained(f"{args.output_dir}/whisper-student-final")
    model.processor.save_pretrained(f"{args.output_dir}/whisper-student-final")
    print(f"Student model saved to {args.output_dir}/whisper-student-final")


if __name__ == "__main__":
    main()
