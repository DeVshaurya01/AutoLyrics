"""WER/CER computation with Whisper-standard normalization.

Uses jiwer 3.0.5 API. If upgrading to 4.x, replace jiwer.wer(truth=...)
with jiwer.wer(reference=...).
"""
import jiwer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer


def _build_normalizer(processor):
    """EnglishTextNormalizer requires the english spelling mapping in newer
    transformers versions. Pull it from the tokenizer when available, else
    fall back to an empty mapping (BasicTextNormalizer behavior)."""
    mapping = getattr(processor.tokenizer, "english_spelling_normalizer", {}) or {}
    return EnglishTextNormalizer(mapping)


def make_compute_metrics(processor):
    """Return a compute_metrics function bound to the given processor.

    Returned function is compatible with Seq2SeqTrainer.
    """
    _normalizer = _build_normalizer(processor)

    def compute_metrics(pred) -> dict:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 pad mask with pad token for decoding
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_norm = [_normalizer(p) for p in pred_str]
        label_norm = [_normalizer(l) for l in label_str]

        # Filter empty references after normalization
        pairs = [(r, h) for r, h in zip(label_norm, pred_norm) if r.strip()]
        if not pairs:
            return {"wer": 1.0, "cer": 1.0, "wer_ortho": 1.0, "cer_ortho": 1.0}

        refs, hyps = zip(*pairs)
        wer_norm = jiwer.wer(list(refs), list(hyps))
        cer_norm = jiwer.cer(list(refs), list(hyps))

        pairs_ortho = [(r, h) for r, h in zip(label_str, pred_str) if r.strip()]
        if not pairs_ortho:
            wer_ortho, cer_ortho = 1.0, 1.0
        else:
            refs_ortho, hyps_ortho = zip(*pairs_ortho)
            wer_ortho = jiwer.wer(list(refs_ortho), list(hyps_ortho))
            cer_ortho = jiwer.cer(list(refs_ortho), list(hyps_ortho))

        return {
            "wer": round(wer_norm, 4),
            "cer": round(cer_norm, 4),
            "wer_ortho": round(wer_ortho, 4),
            "cer_ortho": round(cer_ortho, 4),
        }

    return compute_metrics
