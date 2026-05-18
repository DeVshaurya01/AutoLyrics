"""Tests for WER/CER computation."""
import pytest
import jiwer


def test_jiwer_wer_basic():
    """jiwer 3.0.5: wer(["hello world"], ["hello duck"]) == 0.5"""
    result = jiwer.wer(["hello world"], ["hello duck"])
    assert abs(result - 0.5) < 1e-6


def test_jiwer_wer_perfect():
    assert jiwer.wer(["hello world"], ["hello world"]) == 0.0


def test_jiwer_wer_all_wrong():
    assert jiwer.wer(["hello world"], ["foo bar"]) == 1.0


def test_jiwer_cer_basic():
    result = jiwer.cer(["abc"], ["aXc"])
    assert 0.0 < result <= 1.0


def test_compute_metrics_shape():
    """make_compute_metrics returns a dict with the 4 expected keys."""
    from unittest.mock import MagicMock
    import numpy as np
    from autolyrics.evaluation.metrics import make_compute_metrics

    processor = MagicMock()
    processor.tokenizer.pad_token_id = 50256
    processor.tokenizer.batch_decode.side_effect = lambda ids, **kw: ["hello world"] * len(ids)

    compute_metrics = make_compute_metrics(processor)

    pred = MagicMock()
    pred.predictions = np.array([[1, 2, 3]])
    pred.label_ids = np.array([[1, 2, 3]])

    result = compute_metrics(pred)
    assert set(result.keys()) == {"wer", "cer", "wer_ortho", "cer_ortho"}


def test_compute_metrics_empty_refs():
    """Empty references after normalization should return 1.0 sentinel."""
    from unittest.mock import MagicMock
    import numpy as np
    from autolyrics.evaluation.metrics import make_compute_metrics

    processor = MagicMock()
    processor.tokenizer.pad_token_id = 50256
    # Both ref and hyp are empty strings → normalizer produces ""
    processor.tokenizer.batch_decode.side_effect = lambda ids, **kw: [""] * len(ids)

    compute_metrics = make_compute_metrics(processor)
    pred = MagicMock()
    pred.predictions = np.array([[1]])
    pred.label_ids = np.array([[1]])

    result = compute_metrics(pred)
    assert result["wer"] == 1.0
