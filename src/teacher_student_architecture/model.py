"""
WhisperKnowledgeDistillation: PyTorch Lightning module for Knowledge Distillation
from a fine-tuned Whisper Large V3 (Teacher) to a smaller Whisper model (Student).

The Teacher is frozen (no gradients). Only the Student is trained.
The Student learns from two signals:
  1. Hard Loss (CE): Cross-entropy between Student predictions and ground-truth text.
  2. Soft Loss (KL): KL Divergence between Teacher and Student logits,
     softened by a temperature hyperparameter T.

Total Loss = α * Soft_Loss + (1 - α) * Hard_Loss

Designed for BSC MareNostrum:
  - 2 nodes × 4 GPUs each = 8 GPUs total
  - Full FP32 precision
"""

import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from torch.optim import AdamW
import evaluate


class WhisperKnowledgeDistillation(pl.LightningModule):
    def __init__(
        self,
        teacher_model_path: str,
        student_model_path: str,
        processor_path: str = None,
        temperature: float = 2.0,
        alpha: float = 0.7,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: int = 50000,
    ):
        """
        Parameters
        ----------
        teacher_model_path : str
            Path to the fine-tuned Whisper Large V3 checkpoint (Teacher).
        student_model_path : str
            Path or HuggingFace ID for the smaller Whisper model (Student).
            Examples: "openai/whisper-small", "openai/whisper-base", "openai/whisper-tiny"
        processor_path : str, optional
            Path to the WhisperProcessor. Defaults to student_model_path.
        temperature : float
            Temperature for softening the logits distribution.
            Higher T → softer probabilities → more "dark knowledge" transferred.
            Typical values: 1.0 – 5.0. Default: 2.0
        alpha : float
            Weight for the soft loss (KL Divergence).
            Total Loss = α * Soft_Loss + (1 - α) * Hard_Loss.
            Higher α → Student relies more on Teacher's knowledge.
            Default: 0.7
        learning_rate : float
            Learning rate for the Student optimizer.
        weight_decay : float
            Weight decay for AdamW optimizer.
        warmup_steps : int
            Number of linear warmup steps before cosine decay.
        max_steps : int
            Total training steps for scheduler planning.
        """
        super().__init__()
        self.save_hyperparameters()

        # ─── Load Teacher (frozen, eval mode) ─────────────────────────
        self.teacher = WhisperForConditionalGeneration.from_pretrained(
            teacher_model_path
        )
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # ─── Load Student (trainable) ─────────────────────────────────
        self.student = WhisperForConditionalGeneration.from_pretrained(
            student_model_path
        )
        self.student.config.forced_decoder_ids = None
        self.student.config.suppress_tokens = []

        # ─── Processor (for decoding during validation) ───────────────
        proc_path = processor_path or student_model_path
        self.processor = WhisperProcessor.from_pretrained(proc_path)

        # ─── Metrics ──────────────────────────────────────────────────
        self.wer_metric = evaluate.load("wer")

        # ─── KD Hyperparameters ───────────────────────────────────────
        self.temperature = temperature
        self.alpha = alpha

        # ─── Loss function ────────────────────────────────────────────
        self.kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    # ─── Forward ──────────────────────────────────────────────────────

    def forward(self, input_features, labels=None):
        """Forward pass through the Student model only."""
        return self.student(input_features=input_features, labels=labels)

    # ─── Training Step ────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        input_features = batch["input_features"]
        labels = batch["labels"]

        # 1. Student forward pass (with gradients)
        student_outputs = self.student(
            input_features=input_features, labels=labels
        )
        student_logits = student_outputs.logits  # (B, seq_len, vocab_size)
        hard_loss = student_outputs.loss         # Standard CE loss

        # 2. Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_features=input_features, labels=labels
            )
            teacher_logits = teacher_outputs.logits  # (B, seq_len, vocab_size)

        # 3. Align sequence lengths (Teacher and Student may have different seq_len)
        min_seq_len = min(student_logits.size(1), teacher_logits.size(1))
        student_logits_aligned = student_logits[:, :min_seq_len, :]
        teacher_logits_aligned = teacher_logits[:, :min_seq_len, :]

        # 4. Compute Soft Loss (KL Divergence with temperature scaling)
        #    - Divide logits by T to "soften" the probability distribution
        #    - Student: log_softmax (KLDivLoss expects log-probabilities as input)
        #    - Teacher: softmax (KLDivLoss expects probabilities as target)
        T = self.temperature
        student_soft = F.log_softmax(student_logits_aligned / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits_aligned / T, dim=-1)

        # KL Divergence, scaled by T² (standard KD formulation from Hinton et al.)
        soft_loss = self.kl_loss_fn(student_soft, teacher_soft) * (T ** 2)

        # 5. Combine losses
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        # 6. Logging
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_hard_loss", hard_loss, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train_soft_loss", soft_loss, on_step=True, on_epoch=False, sync_dist=True)

        return total_loss

    # ─── Validation Step ──────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        input_features = batch["input_features"]
        labels = batch["labels"]

        # 1. Student forward pass
        student_outputs = self.student(
            input_features=input_features, labels=labels
        )
        hard_loss = student_outputs.loss
        self.log("val_loss", hard_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # 2. Generate predictions from the Student for WER
        generated_ids = self.student.generate(input_features=input_features)
        pred_str = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Decode ground truth labels
        labels_clean = labels.clone()
        labels_clean[labels_clean == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(labels_clean, skip_special_tokens=True)

        # 3. Compute WER
        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)
        self.log("val_wer", wer, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"val_loss": hard_loss, "val_wer": wer}

    # ─── Optimizer & Scheduler ────────────────────────────────────────

    def configure_optimizers(self):
        # Only optimize Student parameters (Teacher is frozen)
        optimizer = AdamW(
            self.student.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        warmup_steps = self.hparams.warmup_steps
        max_steps = self.hparams.max_steps

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, max_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]
