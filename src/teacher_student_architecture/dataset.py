"""
This module contains the DataModule for the Euskera Transcriptor task.
It handles loading, splitting, scaling, and creating data loaders for
the dataset.
"""

__docformat__ = "numpy"

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import WhisperProcessor
from dataclasses import dataclass
from typing import Any, Dict, List, Union

class EuskeraTranscriptorDataset(Dataset):
    def __init__(self, use_clean_data: bool, data_dir_clean, data_dir_noisy, processor: WhisperProcessor):
        self.use_clean_data = use_clean_data
        self.data_dir_clean = data_dir_clean
        self.data_dir_noisy = data_dir_noisy

        self.data_dir = ""
        self.processor = processor
        self.audio_paths = []
        self.transcriptions = []
        self._load_data()
    
    def _load_data(self):
        if self.use_clean_data:
            self.data_dir = self.data_dir_clean
        else:
            self.data_dir = self.data_dir_noisy
        
        import os
        import csv
        import glob
        
        # Buscar el archivo .tsv en el directorio
        tsv_files = glob.glob(os.path.join(self.data_dir, "*.tsv"))
        if not tsv_files:
            raise FileNotFoundError(f"No se encontró un archivo .tsv en {self.data_dir}")
        
        tsv_path = tsv_files[0]
        print(f"Leyendo dataset desde: {tsv_path}")
        
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if len(row) < 2: continue
                
                # Saltar cabecera si la primera fila parece serlo
                if i == 0 and ("file" in row[0].lower() or "name" in row[0].lower() or "path" in row[0].lower()):
                    continue
                
                audio_filename = row[0]
                transcription = row[1]
                
                # Construir la ruta completa al archivo de audio
                full_audio_path = os.path.join(self.data_dir, audio_filename)
                
                self.audio_paths.append(full_audio_path)
                self.transcriptions.append(transcription)
        
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        # 1. Load audio
        audio, sample_rate = torchaudio.load(self.audio_paths[idx])
        
        # Whisper expects audios always at 16kHz (16000 Hz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio = resampler(audio)
            
        # torchaudio loads with shape (Channels, Time). Whisper processes a single channel in 1D.
        audio = audio.squeeze(0).numpy()
        
        # 2. Get text
        transcription = self.transcriptions[idx]
        
        # 3. Convert audio to Mel spectrogram, and text to token IDs
        input_features = self.processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
        labels = self.processor.tokenizer(transcription).input_ids
        
        return {
            "input_features": input_features, # Mel spectrograms
            "labels": labels                  # Word tokens
        }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    This Data Collator will process the list of dictionaries that __getitem__ returns.
    It will intelligently pad the audios on one side and the texts on the other, 
    adjusting their padding to generate blocks of the same size.
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Separate inputs (audio) and labels (text) as they have different sizes and padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Send to the processor to equalize the sizes in the audio block ("input_features")
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Equalize the size of the texts by adding padding ("labels")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Important: Replace the text padding from 0 to -100 
        # so that PyTorch loss ignores it and does not penalize the padding
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch


class EuskeraTranscriptorDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, model_name_or_path, batch_size=8, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name_or_path = model_name_or_path
        
        # Initialize the official processor
        self.processor = WhisperProcessor.from_pretrained(
            self.model_name_or_path, 
            task="transcribe", 
            # language="basque" # <- You can enable this if you are only going to use Basque
        )
        
        # Here we define the dynamic way in which the batches will be grouped
        self.collate_fn = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = EuskeraTranscriptorDataset(self.data_dir, processor=self.processor)
            self.val_dataset = EuskeraTranscriptorDataset(self.data_dir, processor=self.processor)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=self.collate_fn, #   Apply collation in each batch
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            collate_fn=self.collate_fn,
            num_workers=self.num_workers
        )