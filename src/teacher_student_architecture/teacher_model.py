import os
import torch
import jiwer
import numpy as np
import pytorch_lightning as pl
import io
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    get_linear_schedule_with_warmup
)
class TeacherModel(pl.LightningModule):
    def __init__(self, model_path, learning_rate, weight_decay, warmup_steps, max_steps):
        super().__init__()
        self.save_hyperparameters()
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
        self.processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
        
        self.model.generation_config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="basque", task="transcribe")
        self.model.generation_config.suppress_tokens = []
        
        # FIX CRITICO 2: Desactivar cache para evitar colisiones de memoria en el forward
        self.model.config.use_cache = False
        
        # FIX CRITICO 3: Encoder congelado por seguridad maxima al arrancar
        self.model.model.encoder.requires_grad_(False)

    def forward(self, input_features, labels):
        return self.model(input_features=input_features, labels=labels)

    def training_step(self, batch, batch_idx):
        is_master = self.global_rank == 0
        input_features = batch["input_features"]
        labels = batch["labels"]
        errores_collator = batch.get("num_errores", 0)
        
        total_etiquetas = labels.numel()
        vacias = (labels == -100).sum().item()
        
        # SUPER DEBUG PRINT
        if is_master and (batch_idx < 5 or batch_idx % 50 == 0):
            print(f"\n--- DEBUG STEP {batch_idx} ---")
            print(f" [GPU 0] Errores lectura DataCollator: {errores_collator}")
            print(f" [GPU 0] Features Shape: {input_features.shape}, Labels Shape: {labels.shape}")
            print(f" [GPU 0] Input Features Max: {input_features.max().item():.4f}, Min: {input_features.min().item():.4f}")
            print(f" [GPU 0] Etiquetas a -100 (vacias): {vacias} de {total_etiquetas} totales")

        porcentaje_basura = vacias / total_etiquetas
        
        if porcentaje_basura > 0.75:
            if is_master: 
                print(f" [ALERTA GPU 0] Batch {batch_idx} toxico ({(porcentaje_basura*100):.1f}% basura). Saltando para evitar NaN.")
            # Dummy loss simbolico para mantener las GPUs sincronizadas
            dummy_loss = sum([p.sum() for p in self.parameters() if p.requires_grad]) * 0.0 + 1.5
            return dummy_loss

        # Convertimos los audios al mismo formato que el modelo para que no de RuntimeError
        capa_conv1_dtype = self.model.model.encoder.conv1.weight.dtype
        input_features = input_features.to(capa_conv1_dtype)

        # Forward real
        outputs = self(input_features=input_features, labels=labels)
        loss = outputs.loss

        if is_master and (batch_idx < 10 or batch_idx % 10 == 0):
             print(f" [GPU 0 | Step {batch_idx}] Raw Loss Red: {loss.item():.5f}")

        # Escudo final contra NaN
        if torch.isnan(loss) or torch.isinf(loss):
            if is_master: print(f" [ERROR GPU 0] Loss NaN en step {batch_idx}. Aplicando Dummy.")
            dummy_loss = sum([p.sum() for p in self.parameters() if p.requires_grad]) * 0.0 + 1.5
            return dummy_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    def validation_step(self, batch, batch_idx):
        input_features = batch["input_features"]
        labels = batch["labels"]

        capa_conv1_dtype = self.model.model.encoder.conv1.weight.dtype
        input_features = input_features.to(capa_conv1_dtype)

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, eps=1e-5, betas=(0.9, 0.98))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.max_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
