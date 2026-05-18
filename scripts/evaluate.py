"""Entry point: baseline + fine-tuned model evaluation on test split."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import torch
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
import jiwer

from autolyrics.models.load import load_baseline, load_finetuned
from autolyrics.evaluation.report import generate_report
from autolyrics.utils.paths import DATA_DIR, OUTPUTS_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_normalizer(processor):
    mapping = getattr(processor.tokenizer, "english_spelling_normalizer", {}) or {}
    return EnglishTextNormalizer(mapping)


_normalizer = None  # initialized in main() once processor is loaded


def run_inference(model, processor, dataset, num_beams: int = 1) -> list[dict]:
    """Generate transcriptions for all test samples."""
    model.eval()
    predictions = []

    for sample in dataset:
        input_features = torch.tensor(
            sample["input_features"]
        ).to(dtype=model.dtype, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            generated = model.generate(
                input_features,
                language="english",
                task="transcribe",
                num_beams=num_beams,
                max_new_tokens=225,
            )

        hyp = processor.tokenizer.decode(generated[0], skip_special_tokens=True)

        # Reconstruct reference from label_ids
        label_ids = [t for t in sample["labels"] if t != -100]
        ref = processor.tokenizer.decode(label_ids, skip_special_tokens=True)

        predictions.append({
            "ref": ref,
            "hyp": hyp,
            "singer_id": sample.get("singer_id", "unknown"),
            "song_id": sample.get("song_id", "unknown"),
            "mode": sample.get("mode", "unknown"),
        })

    return predictions


def compute_wer(predictions: list[dict]) -> float:
    refs = [_normalizer(p["ref"]) for p in predictions]
    hyps = [_normalizer(p["hyp"]) for p in predictions]
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return 1.0
    r_list, h_list = zip(*pairs)
    return jiwer.wer(list(r_list), list(h_list))


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline and fine-tuned models")
    parser.add_argument(
        "--dataset",
        default=str(DATA_DIR / "processed" / "hf_dataset"),
    )
    parser.add_argument(
        "--adapter_path",
        default=str(OUTPUTS_DIR / "checkpoints" / "final_adapter"),
        help="Path to saved LoRA adapter (skip with --baseline_only)",
    )
    parser.add_argument(
        "--output_dir",
        default=str(OUTPUTS_DIR / "reports"),
    )
    parser.add_argument(
        "--baseline_only", action="store_true",
        help="Only evaluate baseline (no fine-tuned model needed)",
    )
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    print("=" * 60)
    print("AutoLyrics — Evaluation")
    print("=" * 60)

    ds = load_from_disk(args.dataset)
    test_ds = ds["test"]

    # Baseline
    print("\n[1/2] Evaluating baseline Whisper-small...")
    base_model, processor = load_baseline()
    global _normalizer
    _normalizer = _make_normalizer(processor)
    base_preds = run_inference(base_model, processor, test_ds, args.num_beams)
    base_wer = compute_wer(base_preds)
    print(f"      Baseline WER (normalized): {base_wer:.4f}")

    generate_report(base_preds, str(args.output_dir) + "/baseline")

    if not args.baseline_only:
        # Fine-tuned
        print("\n[2/2] Evaluating fine-tuned Whisper-small + LoRA...")
        ft_model, ft_processor = load_finetuned(args.adapter_path)
        ft_preds = run_inference(ft_model, ft_processor, test_ds, args.num_beams)
        ft_wer = compute_wer(ft_preds)
        print(f"      Fine-tuned WER (normalized): {ft_wer:.4f}")

        generate_report(ft_preds, str(args.output_dir) + "/finetuned", baseline_wer=base_wer)

        rel = (base_wer - ft_wer) / base_wer * 100
        print(f"\n[RESULT] Relative WER reduction: {rel:.1f}%")
        print(f"         Target: >15% | {'PASS' if rel > 15 else 'BELOW TARGET'}")

    print(f"\n[DONE] Reports saved to {args.output_dir}")


if __name__ == "__main__":
    main()
