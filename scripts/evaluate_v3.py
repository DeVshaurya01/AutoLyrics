"""Evaluation v3 — per-chunk WER on the test split.

Uses the already-prepared test chunks (from prepare_data_v3.py) directly.
Each chunk has properly aligned audio + canonical text. We run baseline and
fine-tuned models on every chunk and compute WER over all chunks together.

This matches the training distribution exactly (same chunk length, same
text format) so the LoRA's learned improvements show through cleanly,
without full-song chunk-stitching introducing noise.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from pathlib import Path
import torch
import jiwer
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from peft import PeftModel

MODEL_ID = "openai/whisper-small"
ADAPTER = Path("outputs/checkpoints/final_adapter")
OUT_DIR = Path("outputs/reports/eval_v3")
DATASET = Path("data/processed/hf_dataset")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def transcribe_chunk(model, processor, input_features):
    feats = torch.tensor(input_features).to(dtype=model.dtype, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        ids = model.generate(
            feats, language="english", task="transcribe",
            num_beams=5, no_repeat_ngram_size=3, max_new_tokens=225,
        )
    return processor.tokenizer.decode(ids[0], skip_special_tokens=True).strip()


def eval_model(name, model, processor, test_ds, normalizer):
    refs, hyps, rows = [], [], []
    for i, sample in enumerate(test_ds):
        hyp = transcribe_chunk(model, processor, sample["input_features"])
        label_ids = [t for t in sample["labels"] if t != -100]
        ref = processor.tokenizer.decode(label_ids, skip_special_tokens=True).strip()
        refs.append(ref)
        hyps.append(hyp)
        rows.append({
            "singer": sample.get("singer_id", "?"),
            "mode": sample.get("mode", "?"),
            "song_id": sample.get("song_id", "?"),
            "ref": ref, "hyp": hyp,
        })
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(test_ds)} chunks done")

    rn = [normalizer(r) for r in refs]
    hn = [normalizer(h) for h in hyps]
    paired = [(r, h) for r, h in zip(rn, hn) if r.strip()]
    r_list, h_list = zip(*paired)
    overall = jiwer.wer(list(r_list), list(h_list))

    sung_pairs = [(rn[i], hn[i]) for i in range(len(rows)) if rows[i]["mode"] == "sing" and rn[i].strip()]
    read_pairs = [(rn[i], hn[i]) for i in range(len(rows)) if rows[i]["mode"] == "read" and rn[i].strip()]
    sung_wer = jiwer.wer([p[0] for p in sung_pairs], [p[1] for p in sung_pairs]) if sung_pairs else None
    read_wer = jiwer.wer([p[0] for p in read_pairs], [p[1] for p in read_pairs]) if read_pairs else None

    print(f"\n[{name}]")
    print(f"  Overall WER: {overall:.4f}  on {len(paired)} chunks")
    if sung_wer is not None:
        print(f"  Sung   WER: {sung_wer:.4f}  ({len(sung_pairs)})")
    if read_wer is not None:
        print(f"  Spoken WER: {read_wer:.4f}  ({len(read_pairs)})")

    return {"overall": overall, "sung": sung_wer, "spoken": read_wer, "rows": rows}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading test split from {DATASET}...")
    ds = load_from_disk(str(DATASET))
    test_ds = ds["test"]
    print(f"  {len(test_ds)} test chunks")

    processor = WhisperProcessor.from_pretrained(
        MODEL_ID, language="English", task="transcribe"
    )
    normalizer = EnglishTextNormalizer(
        getattr(processor.tokenizer, "english_spelling_normalizer", {}) or {}
    )

    # Baseline
    print("\nLoading baseline...")
    base = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE
    ).to(DEVICE).eval()
    base_res = eval_model("Baseline", base, processor, test_ds, normalizer)
    del base
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Fine-tuned
    ft_res = None
    if ADAPTER.exists():
        print("\nLoading fine-tuned (LoRA)...")
        ft_base = WhisperForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=DTYPE
        )
        ft = PeftModel.from_pretrained(ft_base, str(ADAPTER))
        ft = ft.to(DEVICE).eval()
        ft.config.use_cache = True
        ft_res = eval_model("Fine-tuned (LoRA)", ft, processor, test_ds, normalizer)
    else:
        print(f"\n[INFO] No adapter at {ADAPTER}, skipping fine-tuned eval.")

    out = {
        "model_id": MODEL_ID,
        "baseline": {k: v for k, v in base_res.items() if k != "rows"},
        "finetuned": {k: v for k, v in ft_res.items() if k != "rows"} if ft_res else None,
    }
    if ft_res:
        rel_overall = (base_res["overall"] - ft_res["overall"]) / base_res["overall"] * 100
        rel_sung = ((base_res["sung"] - ft_res["sung"]) / base_res["sung"] * 100
                    if base_res["sung"] and ft_res["sung"] else None)
        rel_read = ((base_res["spoken"] - ft_res["spoken"]) / base_res["spoken"] * 100
                    if base_res["spoken"] and ft_res["spoken"] else None)
        out["relative_reduction_pct"] = {
            "overall": round(rel_overall, 2),
            "sung": round(rel_sung, 2) if rel_sung is not None else None,
            "spoken": round(rel_read, 2) if rel_read is not None else None,
        }
        print(f"\n[RESULT] Relative WER reduction:")
        print(f"  Overall: {rel_overall:.2f}%")
        if rel_sung is not None:
            print(f"  Sung:    {rel_sung:.2f}%")
        if rel_read is not None:
            print(f"  Spoken:  {rel_read:.2f}%")
        print(f"\n  Target: >=15% (relative reduction)")

    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved to {OUT_DIR/'results.json'}")


if __name__ == "__main__":
    main()
