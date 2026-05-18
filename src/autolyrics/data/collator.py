"""Speech-Seq2Seq data collator with optional SpecAugment."""
from dataclasses import dataclass
from typing import Any
import torch
import torchaudio.transforms as T


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    apply_specaug: bool = False  # Enable for training batches only

    def __call__(self, features: list[dict]) -> dict:
        # Pad input features (mel spectrograms)
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        if self.apply_specaug:
            batch["input_features"] = self._specaugment(batch["input_features"])

        # Pad labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Mask padding tokens with -100 so they are ignored in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Strip BOS token if prepended by the tokenizer
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

    @staticmethod
    def _specaugment(mel: torch.Tensor) -> torch.Tensor:
        """2× time masks + 2× freq masks (SpecAugment)."""
        # mel: (batch, n_mels=80, time=3000)
        for _ in range(2):
            mel = T.TimeMasking(time_mask_param=40)(mel)
        for _ in range(2):
            mel = T.FrequencyMasking(freq_mask_param=10)(mel)
        return mel
