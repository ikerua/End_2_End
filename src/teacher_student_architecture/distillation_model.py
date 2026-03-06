"""
distillation_model.py - Knowledge Distillation Lightning Module
Transfers knowledge from a fine-tuned Whisper Large-V3 (Teacher)
to a Whisper Base (Student) for Basque transcription.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import jiwer
import pytorch_lightning as pl

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    get_linear_schedule_with_warmup,
)


class WhisperDistillationModule(pl.LightningModule):
    """
    Knowledge Distillation module that trains a Whisper Base student
    using soft targets from a frozen Whisper Large-V3 teacher loaded
    from a PyTorch Lightning checkpoint.

    Parameters
    ----------
    teacher_model_path : str
        Path to the base Whisper Large-V3 HuggingFace model directory
        (needed to instantiate the architecture and processor).
    teacher_ckpt_path : str
        Path to the PyTorch Lightning ``.ckpt`` file containing the
        fine-tuned Teacher weights.
    student_model_name : str
        HuggingFace model identifier or local path for the Whisper Base
        student (e.g. ``"openai/whisper-base"``).
    temperature : float
        Softmax temperature for distillation (default: 2.0).
    alpha : float
        Weight balance: ``alpha * KD_loss + (1-alpha) * CE_loss``
        (default: 0.5).
    learning_rate : float
        Peak learning rate for the student (default: 5e-5).
    weight_decay : float
        AdamW weight decay (default: 0.05).
    warmup_steps : int
        Linear warmup steps for the scheduler (default: 500).
    max_steps : int
        Total training steps for the scheduler (default: 50000).
    freeze_student_encoder : bool
        Whether to freeze the student encoder (default: True).
        Freezing the encoder is common for distillation since the
        encoder's mel-spectrogram representations transfer well.
    """

    def __init__(
        self,
        teacher_model_path: str,
        teacher_ckpt_path: str,
        student_model_name: str = "openai/whisper-base",
        temperature: float = 2.0,
        alpha: float = 0.5,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.05,
        warmup_steps: int = 500,
        max_steps: int = 50000,
        freeze_student_encoder: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ----------------------------------------------------------
        # 1. TEACHER: Load architecture + fine-tuned checkpoint
        # ----------------------------------------------------------
        print("[Distillation] Loading Teacher architecture from:", teacher_model_path)
        self.teacher_model = WhisperForConditionalGeneration.from_pretrained(
            teacher_model_path, local_files_only=True
        )
        self.teacher_processor = WhisperProcessor.from_pretrained(
            teacher_model_path, local_files_only=True
        )

        # Load the fine-tuned weights from the Lightning checkpoint
        print("[Distillation] Loading Teacher weights from checkpoint:", teacher_ckpt_path)
        ckpt = torch.load(teacher_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]

        # The checkpoint keys are prefixed with "model." (from the
        # TeacherModel LightningModule wrapping), so we strip that
        # prefix to match the HuggingFace model keys.
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            # "model.model.encoder..." -> "model.encoder..."
            if key.startswith("model."):
                new_key = key[len("model."):]
                cleaned_state_dict[new_key] = value
            else:
                cleaned_state_dict[key] = value

        missing, unexpected = self.teacher_model.load_state_dict(cleaned_state_dict, strict=False)
        if missing:
            print(f"[Distillation] WARNING — Teacher missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"[Distillation] WARNING — Teacher unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

        # Freeze the Teacher completely and set to eval mode
        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()
        self.teacher_model.config.use_cache = False

        # ----------------------------------------------------------
        # 2. STUDENT: Load Whisper Base
        # ----------------------------------------------------------
        print("[Distillation] Loading Student model:", student_model_name)
        self.student_model = WhisperForConditionalGeneration.from_pretrained(
            student_model_name
        )
        self.student_processor = WhisperProcessor.from_pretrained(
            student_model_name
        )

        # Configure student for Basque transcription
        self.student_model.generation_config.forced_decoder_ids = (
            self.student_processor.get_decoder_prompt_ids(
                language="basque", task="transcribe"
            )
        )
        self.student_model.generation_config.suppress_tokens = []
        self.student_model.config.use_cache = False

        if freeze_student_encoder:
            print("[Distillation] Freezing student encoder.")
            self.student_model.model.encoder.requires_grad_(False)

        # ----------------------------------------------------------
        # 3. STUDENT FEATURE EXTRACTOR (80 mel bands)
        # ----------------------------------------------------------
        # The DataCollator uses the Teacher's processor (128 mel bands).
        # The Student expects 80 mel bands.  We keep the student's
        # feature extractor so we can re-extract features at 80 bands
        # inside the training_step if needed.
        self.student_feature_extractor = WhisperFeatureExtractor.from_pretrained(
            student_model_name
        )

        # Quick sanity checks
        self.teacher_mel_bands = self.teacher_model.config.num_mel_bins  # 128
        self.student_mel_bands = self.student_model.config.num_mel_bins  # 80
        print(
            f"[Distillation] Teacher mel bands: {self.teacher_mel_bands}, "
            f"Student mel bands: {self.student_mel_bands}"
        )

        # ----------------------------------------------------------
        # 4. PROJECTION LAYER for mel-band adaptation (128 -> 80)
        # ----------------------------------------------------------
        # Instead of re-extracting features (expensive), we learn a
        # lightweight linear projection from 128 mel bands to 80.
        # This preserves the temporal dimension (3000 frames).
        if self.teacher_mel_bands != self.student_mel_bands:
            print(
                f"[Distillation] Adding mel-band projection layer: "
                f"{self.teacher_mel_bands} -> {self.student_mel_bands}"
            )
            self.mel_projection = nn.Linear(
                self.teacher_mel_bands, self.student_mel_bands, bias=False
            )
            # Initialize close to identity-like (truncated top rows)
            with torch.no_grad():
                nn.init.eye_(self.mel_projection.weight)
        else:
            self.mel_projection = None

        # ----------------------------------------------------------
        # 5. LOGIT PROJECTION (vocab alignment)
        # ----------------------------------------------------------
        # Teacher vocab size may differ from Student vocab size.
        teacher_vocab = self.teacher_model.config.vocab_size
        student_vocab = self.student_model.config.vocab_size
        if teacher_vocab != student_vocab:
            print(
                f"[Distillation] Adding vocab projection layer: "
                f"teacher {teacher_vocab} -> student {student_vocab}"
            )
            self.logit_projection = nn.Linear(
                teacher_vocab, student_vocab, bias=False
            )
        else:
            self.logit_projection = None

        # Distillation loss
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    # ==============================================================
    # FORWARD
    # ==============================================================
    def forward(self, input_features, labels):
        """Forward pass through the student model only."""
        return self.student_model(input_features=input_features, labels=labels)

    # ==============================================================
    # ADAPT INPUT FEATURES (128 mel -> 80 mel)
    # ==============================================================
    def _adapt_features_for_student(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Convert 128-mel-band features to 80-mel-band features
        for the student model.

        Parameters
        ----------
        input_features : torch.Tensor
            Shape ``(batch, 128, time_steps)``

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 80, time_steps)``
        """
        if self.mel_projection is None:
            return input_features

        # (B, 128, T) -> (B, T, 128) -> Linear -> (B, T, 80) -> (B, 80, T)
        x = input_features.permute(0, 2, 1)        # (B, T, 128)
        x = self.mel_projection(x)                  # (B, T, 80)
        x = x.permute(0, 2, 1)                      # (B, 80, T)
        return x

    # ==============================================================
    # TRAINING STEP
    # ==============================================================
    def training_step(self, batch, batch_idx):
        is_master = self.global_rank == 0
        input_features = batch["input_features"]        # (B, 128, T)
        labels = batch["labels"]
        errores_collator = batch.get("num_errores", 0)

        total_etiquetas = labels.numel()
        vacias = (labels == -100).sum().item()
        porcentaje_basura = vacias / total_etiquetas

        if is_master and (batch_idx < 5 or batch_idx % 50 == 0):
            print(f"\n--- DISTILL STEP {batch_idx} ---")
            print(f"  Errores DataCollator: {errores_collator}")
            print(f"  Features: {input_features.shape}, Labels: {labels.shape}")
            print(f"  Etiquetas vacías (-100): {vacias}/{total_etiquetas}")

        # Skip toxic batches (>75% padding)
        if porcentaje_basura > 0.75:
            if is_master:
                print(f"  [ALERTA] Batch {batch_idx} tóxico ({porcentaje_basura*100:.1f}% basura). Saltando.")
            dummy_loss = sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0 + 1.5
            return dummy_loss

        # Cast to match model dtype
        teacher_dtype = self.teacher_model.model.encoder.conv1.weight.dtype
        student_dtype = self.student_model.model.encoder.conv1.weight.dtype

        teacher_features = input_features.to(teacher_dtype)
        student_features = self._adapt_features_for_student(input_features).to(student_dtype)

        # ----- TEACHER FORWARD (no grad) -----
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_features=teacher_features, labels=labels
            )
            teacher_logits = teacher_outputs.logits  # (B, seq_len, teacher_vocab)

        # ----- STUDENT FORWARD -----
        student_outputs = self.student_model(
            input_features=student_features, labels=labels
        )
        student_logits = student_outputs.logits  # (B, seq_len, student_vocab)
        ce_loss = student_outputs.loss

        # ----- DISTILLATION LOSS -----
        T = self.hparams.temperature

        # Align sequence lengths (take min)
        min_seq_len = min(teacher_logits.size(1), student_logits.size(1))
        t_logits = teacher_logits[:, :min_seq_len, :]
        s_logits = student_logits[:, :min_seq_len, :]

        # Project teacher logits to student vocab space if needed
        if self.logit_projection is not None:
            t_logits = self.logit_projection(t_logits)

        # Soft targets
        teacher_soft = F.softmax(t_logits / T, dim=-1)
        student_log_soft = F.log_softmax(s_logits / T, dim=-1)

        distill_loss = self.kl_loss(student_log_soft, teacher_soft) * (T ** 2)

        # ----- COMBINED LOSS -----
        alpha = self.hparams.alpha
        loss = alpha * distill_loss + (1.0 - alpha) * ce_loss

        # NaN / Inf shield
        if torch.isnan(loss) or torch.isinf(loss):
            if is_master:
                print(f"  [ERROR] Loss NaN/Inf en step {batch_idx}. Aplicando dummy.")
            dummy_loss = sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0 + 1.5
            return dummy_loss

        if is_master and (batch_idx < 10 or batch_idx % 10 == 0):
            print(
                f"  [Step {batch_idx}] CE: {ce_loss.item():.4f}  "
                f"KD: {distill_loss.item():.4f}  Total: {loss.item():.4f}"
            )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_ce_loss", ce_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_kd_loss", distill_loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    # ==============================================================
    # VALIDATION STEP (Student-only evaluation)
    # ==============================================================
    def validation_step(self, batch, batch_idx):
        input_features = batch["input_features"]
        labels = batch["labels"]

        student_dtype = self.student_model.model.encoder.conv1.weight.dtype
        student_features = self._adapt_features_for_student(input_features).to(student_dtype)

        # Student forward for loss
        student_outputs = self.student_model(
            input_features=student_features, labels=labels
        )
        val_loss = student_outputs.loss
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # Generate predictions with Student
        generated_ids = self.student_model.generate(
            input_features=student_features,
            max_new_tokens=225,
            language="basque",
            task="transcribe",
        )

        decoded_preds = self.student_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        labels_clean = labels.clone()
        labels_clean[labels_clean == -100] = self.student_processor.tokenizer.pad_token_id
        decoded_labels = self.student_processor.batch_decode(
            labels_clean, skip_special_tokens=True
        )

        # Avoid empty strings for jiwer
        decoded_preds = [p if p.strip() else "EMPTY" for p in decoded_preds]
        decoded_labels = [l if l.strip() else "EMPTY" for l in decoded_labels]

        wer = jiwer.wer(decoded_labels, decoded_preds)
        cer = jiwer.cer(decoded_labels, decoded_preds)

        self.log("val_wer", wer, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_cer", cer, on_epoch=True, prog_bar=True, sync_dist=True)

        return val_loss

    # ==============================================================
    # OPTIMIZERS (Student parameters only)
    # ==============================================================
    def configure_optimizers(self):
        # Collect only trainable parameters (student + projection layers)
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        print(
            f"[Distillation] Trainable parameters: "
            f"{sum(p.numel() for p in trainable_params):,}"
        )

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            eps=1e-5,
            betas=(0.9, 0.98),
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.max_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ==============================================================
    # TEACHER ALWAYS IN EVAL MODE
    # ==============================================================
    def on_train_epoch_start(self):
        """Ensure Teacher stays in eval mode after Lightning's
        automatic `model.train()` call."""
        self.teacher_model.eval()
