"""Tests for DataCollatorSpeechSeq2SeqWithPadding."""
import torch
import pytest
from unittest.mock import MagicMock, patch
from autolyrics.data.collator import DataCollatorSpeechSeq2SeqWithPadding


def _make_mock_processor(pad_token_id: int = 50256, bos_id: int = 50258):
    processor = MagicMock()

    # feature_extractor.pad returns a batch dict with padded input_features
    def fake_feat_pad(features, return_tensors):
        n = len(features)
        stacked = torch.stack([torch.tensor(f["input_features"]) for f in features])
        return {"input_features": stacked}

    processor.feature_extractor.pad.side_effect = fake_feat_pad

    # tokenizer.pad returns attention_mask + input_ids
    def fake_tok_pad(label_features, return_tensors):
        ids = [torch.tensor(f["input_ids"]) for f in label_features]
        max_len = max(len(x) for x in ids)
        padded = torch.full((len(ids), max_len), pad_token_id, dtype=torch.long)
        mask = torch.zeros(len(ids), max_len, dtype=torch.long)
        for i, row in enumerate(ids):
            padded[i, : len(row)] = row
            mask[i, : len(row)] = 1
        result = MagicMock()
        result.input_ids = padded
        result.attention_mask = mask
        return result

    processor.tokenizer.pad.side_effect = fake_tok_pad
    return processor, bos_id


def _make_features(n: int = 4, mel_len: int = 3000, n_mels: int = 80, label_lens=None):
    label_lens = label_lens or [5, 7, 3, 6]
    features = []
    for i in range(n):
        features.append({
            "input_features": torch.randn(n_mels, mel_len).tolist(),
            "labels": list(range(1, label_lens[i] + 1)),
        })
    return features


def test_collator_output_keys():
    processor, bos_id = _make_mock_processor()
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=bos_id, apply_specaug=False
    )
    batch = collator(_make_features())
    assert "input_features" in batch
    assert "labels" in batch


def test_collator_label_padding_mask():
    """Padding positions in labels must be -100."""
    processor, bos_id = _make_mock_processor(pad_token_id=50256)
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=bos_id, apply_specaug=False
    )
    batch = collator(_make_features(label_lens=[3, 7, 2, 5]))
    labels = batch["labels"]
    # Positions beyond each sample's real length should be -100
    assert (labels == -100).any(), "Expected some -100 padding tokens in labels"


def test_collator_specaugment_runs():
    processor, bos_id = _make_mock_processor()
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=bos_id, apply_specaug=True
    )
    # Should not raise
    batch = collator(_make_features())
    assert batch["input_features"].shape[1] == 80  # n_mels preserved
