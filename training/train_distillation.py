"""
train_distillation.py - Training Script for Teacher-Student Distillation
Trains a Whisper Base student using soft targets from a fine-tuned
Whisper Large-V3 teacher checkpoint.
"""

import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy

from src.teacher_student_architecture.distillation_model import WhisperDistillationModule
from src.teacher_student_architecture.dataset import WhisperDataModule

# Offline mode for BSC / air-gapped environments
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation: Whisper Large-V3 → Whisper Base"
    )

    # --- Model paths ---
    parser.add_argument(
        "--teacher_model_path", type=str,
        default="/gpfs/projects/ehpc485/tesi681824/transcriptor/modelo_whisper_large_v3",
        help="Path to the HuggingFace Whisper Large-V3 model directory (architecture + processor)."
    )
    parser.add_argument(
        "--teacher_ckpt_path", type=str,
        required=True,
        help="Path to the PyTorch Lightning .ckpt file with fine-tuned Teacher weights."
    )
    parser.add_argument(
        "--student_model_name", type=str,
        default="openai/whisper-base",
        help="HuggingFace model name or local path for the Whisper Base student."
    )

    # --- Data ---
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the HuggingFace dataset on disk.")

    # --- Training hyperparameters ---
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--accumulate_grad_batches", type=int, default=2)

    # --- Distillation hyperparameters ---
    parser.add_argument("--temperature", type=float, default=2.0, help="Softmax temperature for KD.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight: alpha*KD + (1-alpha)*CE.")
    parser.add_argument("--freeze_student_encoder", action="store_true", default=True,
                        help="Freeze the student encoder during training.")
    parser.add_argument("--no_freeze_student_encoder", dest="freeze_student_encoder", action="store_false")

    # --- Hardware ---
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--gpus_per_node", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)

    # --- Output ---
    parser.add_argument("--output_dir", type=str,
                        default="/gpfs/scratch/ehpc485/tesi681824/whisper_distill_out/checkpoints")
    parser.add_argument("--log_dir", type=str,
                        default="/gpfs/scratch/ehpc485/tesi681824/whisper_distill_out/logs")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # =============================================
    # MODEL
    # =============================================
    model = WhisperDistillationModule(
        teacher_model_path=args.teacher_model_path,
        teacher_ckpt_path=args.teacher_ckpt_path,
        student_model_name=args.student_model_name,
        temperature=args.temperature,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        freeze_student_encoder=args.freeze_student_encoder,
    )

    # =============================================
    # DATA  (uses the Teacher processor for 128 mel-band features)
    # =============================================
    datamodule = WhisperDataModule(
        data_dir=args.data_dir,
        model_path=args.teacher_model_path,  # 128 mel bands
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # =============================================
    # CALLBACKS
    # =============================================
    checkpoint_best = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="student-best-epoch={epoch:02d}-wer={val_wer:.4f}",
        monitor="val_wer",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_wer", mode="min", patience=5, verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    csv_logger = CSVLogger(save_dir=args.log_dir, name="distillation_logs")

    # =============================================
    # TRAINER
    # =============================================
    trainer = pl.Trainer(
        num_nodes=args.num_nodes,
        devices=args.gpus_per_node,
        strategy=DDPStrategy(find_unused_parameters=True),  # projection layers may be unused on some steps
        accelerator="gpu",
        precision="32-true",
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
    print(" KNOWLEDGE DISTILLATION: WHISPER LARGE-V3 → WHISPER BASE")
    print(f" Temperature: {args.temperature}  |  Alpha: {args.alpha}")
    print(f" Student encoder frozen: {args.freeze_student_encoder}")
    print("================================================================")

    trainer.fit(model, datamodule=datamodule)

    # Save the final Student model in HuggingFace format
    print("Guardando el modelo Student final...")
    final_path = os.path.join(args.output_dir, "whisper-base-euskera-distilled")
    model.student_model.save_pretrained(final_path)
    model.student_processor.save_pretrained(final_path)
    print(f"Modelo Student guardado en: {final_path}")
    print("Proceso terminado.")


if __name__ == "__main__":
    main()
