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
    def __init__(
        self,
        teacher_model_path: str,
        teacher_ckpt_path: str,
        student_model_name: str = "/gpfs/projects/ehpc485/tesi681824/transcriptor/modelo_whisper_base",
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

        print("[Distillation] Loading Teacher weights from checkpoint:", teacher_ckpt_path)
        ckpt = torch.load(teacher_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_key = key[len("model."):]
                cleaned_state_dict[new_key] = value
            else:
                cleaned_state_dict[key] = value

        missing, unexpected = self.teacher_model.load_state_dict(cleaned_state_dict, strict=False)
        if missing:
            print(f"[Distillation] WARNING - Teacher missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"[Distillation] WARNING - Teacher unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

        # Freeze the Teacher completely and set to eval mode
        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()
        self.teacher_model.config.use_cache = False

        # ----------------------------------------------------------
        # 2. STUDENT: Load Whisper Base 
        # ----------------------------------------------------------
        print("[Distillation] Loading Student model:", student_model_name)
        self.student_model = WhisperForConditionalGeneration.from_pretrained(
            student_model_name, local_files_only=True
        )
        self.student_processor = WhisperProcessor.from_pretrained(
            student_model_name, local_files_only=True
        )

        # Configure student for Basque transcription
        self.student_model.generation_config.forced_decoder_ids = (
            self.student_processor.get_decoder_prompt_ids(
                language="basque", task="transcribe"
            )
        )
        self.student_model.generation_config.suppress_tokens = []
        self.student_model.config.use_cache = False
        
        # Salvavidas de memoria VRAM para el Student
        self.student_model.gradient_checkpointing_enable()

        if freeze_student_encoder:
            print("[Distillation] Freezing student encoder.")
            self.student_model.model.encoder.requires_grad_(False)

        # ----------------------------------------------------------
        # 3. STUDENT FEATURE EXTRACTOR
        # ----------------------------------------------------------
        self.student_feature_extractor = WhisperFeatureExtractor.from_pretrained(
            student_model_name, local_files_only=True
        )

        self.teacher_mel_bands = self.teacher_model.config.num_mel_bins  
        self.student_mel_bands = self.student_model.config.num_mel_bins  
        print(
            f"[Distillation] Teacher mel bands: {self.teacher_mel_bands}, "
            f"Student mel bands: {self.student_mel_bands}"
        )

        # ----------------------------------------------------------
        # 4. PROJECTION LAYER (128 -> 80)
        # ----------------------------------------------------------
        if self.teacher_mel_bands != self.student_mel_bands:
            print(
                f"[Distillation] Adding mel-band projection layer: "
                f"{self.teacher_mel_bands} -> {self.student_mel_bands}"
            )
            self.mel_projection = nn.Linear(
                self.teacher_mel_bands, self.student_mel_bands, bias=False
            )
            with torch.no_grad():
                nn.init.eye_(self.mel_projection.weight)
        else:
            self.mel_projection = None

    def forward(self, input_features, labels):
        return self.student_model(input_features=input_features, labels=labels)

    def _adapt_features_for_student(self, input_features: torch.Tensor) -> torch.Tensor:
        if self.mel_projection is None:
            return input_features

        x = input_features.permute(0, 2, 1)        # (B, T, 128)
        # Asegurar tipo de datos para FP16 vs FP32
        x = x.to(self.mel_projection.weight.dtype)
        x = self.mel_projection(x)                 # (B, T, 80)
        x = x.permute(0, 2, 1)                     # (B, 80, T)
        return x

    def training_step(self, batch, batch_idx):
        is_master = self.global_rank == 0
        input_features = batch["input_features"]        
        labels = batch["labels"]
        errores_collator = batch.get("num_errores", 0)

        total_etiquetas = labels.numel()
        vacias = (labels == -100).sum().item()
        porcentaje_basura = vacias / total_etiquetas

        if is_master and (batch_idx < 5 or batch_idx % 50 == 0):
            print(f"\n--- DISTILL STEP {batch_idx} ---")
            print(f"  Errores DataCollator: {errores_collator}")
            print(f"  Features: {input_features.shape}, Labels: {labels.shape}")
            print(f"  Etiquetas vacias (-100): {vacias}/{total_etiquetas}")

        if porcentaje_basura > 0.75:
            if is_master:
                print(f"  [ALERTA] Batch {batch_idx} toxico ({porcentaje_basura*100:.1f}% basura). Saltando.")
            dummy_loss = sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0 + 1.5
            return dummy_loss

        teacher_dtype = self.teacher_model.model.encoder.conv1.weight.dtype
        student_dtype = self.student_model.model.encoder.conv1.weight.dtype

        teacher_features = input_features.to(teacher_dtype)
        student_features = self._adapt_features_for_student(input_features).to(student_dtype)

        # ----- TEACHER FORWARD -----
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_features=teacher_features, labels=labels
            )
            teacher_logits = teacher_outputs.logits  

        # ----- STUDENT FORWARD -----
        student_outputs = self.student_model(
            input_features=student_features, labels=labels
        )
        student_logits = student_outputs.logits  
        ce_loss = student_outputs.loss

        # ----- DISTILLATION LOSS (ARREGLO ANTI-NAN) -----
        T = self.hparams.temperature

        # 1. Alinear longitudes de secuencia y vocabulario
        min_seq_len = min(teacher_logits.size(1), student_logits.size(1))
        min_vocab = min(teacher_logits.size(-1), student_logits.size(-1))

        t_logits = teacher_logits[:, :min_seq_len, :min_vocab]
        s_logits = student_logits[:, :min_seq_len, :min_vocab]
        labels_sliced = labels[:, :min_seq_len]

        # 2. Calcular probabilidades SIN mascaras previas para evitar division por cero
        teacher_soft = F.softmax((t_logits / T).float(), dim=-1)
        student_log_soft = F.log_softmax((s_logits / T).float(), dim=-1)

        # 3. Calcular la divergencia KL sin reducir (nos da el error por cada token y clase)
        # Result shape: (Batch, Seq_Len, Vocab)
        distill_loss_unmasked = F.kl_div(student_log_soft, teacher_soft, reduction="none") * (T ** 2)

        # 4. Sumar el error sobre todo el vocabulario para tener el error total por token
        # Result shape: (Batch, Seq_Len)
        distill_loss_per_token = distill_loss_unmasked.sum(dim=-1)

        # 5. Crear la mascara para identificar tokens validos (1.0) y padding (0.0)
        valid_mask = (labels_sliced != -100).float()
        
        # 6. Multiplicar el error por la mascara (borra el error del padding) y promediar
        # clamp(min=1.0) evita division por cero si por un milagro todo el batch es padding
        distill_loss = (distill_loss_per_token * valid_mask).sum() / torch.clamp(valid_mask.sum(), min=1.0)

        # ----- COMBINED LOSS -----
        alpha = self.hparams.alpha
        loss = alpha * distill_loss + (1.0 - alpha) * ce_loss

        if torch.isnan(loss) or torch.isinf(loss):
            if is_master:
                print(f"  [ERROR] Loss NaN/Inf en step {batch_idx}. Aplicando dummy.")
            dummy_loss = sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0 + 1.5
            return dummy_loss

        if is_master and (batch_idx < 10 or batch_idx % 10 == 0):
            print(
                f"  [Step {batch_idx}] CE: {ce_loss.item():.4f}  "
                f"KD: {distill_loss.item():.4f}  Total: {loss.item():.4f}"
            )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_ce_loss", ce_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_kd_loss", distill_loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_features = batch["input_features"]
        labels = batch["labels"]

        student_dtype = self.student_model.model.encoder.conv1.weight.dtype
        student_features = self._adapt_features_for_student(input_features).to(student_dtype)

        student_outputs = self.student_model(
            input_features=student_features, labels=labels
        )
        val_loss = student_outputs.loss
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, sync_dist=True)

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

        decoded_preds = [p if p.strip() else "EMPTY" for p in decoded_preds]
        decoded_labels = [l if l.strip() else "EMPTY" for l in decoded_labels]

        wer = jiwer.wer(decoded_labels, decoded_preds)
        cer = jiwer.cer(decoded_labels, decoded_preds)

        self.log("val_wer", wer, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_cer", cer, on_epoch=True, prog_bar=True, sync_dist=True)

        return val_loss

    def configure_optimizers(self):
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

    def on_train_epoch_start(self):
        self.teacher_model.eval()

