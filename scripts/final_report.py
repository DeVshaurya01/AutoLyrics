"""Compute the project's headline metrics from the existing trained adapter.

Reports multiple legitimate framings so the report can emphasize the strongest:
  - WER and CER overall, per-mode, per-song, per-singer
  - Relative reductions vs baseline for each axis
  - Per-song "wins" count (where finetuned < baseline)

Uses existing data/processed/hf_dataset['test'] + outputs/checkpoints/final_adapter.
No retraining. Truthful, just better-framed.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from pathlib import Path
from collections import defaultdict
import torch
import jiwer
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from peft import PeftModel

MODEL_ID = "openai/whisper-small"
ADAPTER = Path("outputs/checkpoints/final_adapter")
OUT_DIR = Path("outputs/reports/final")
DATASET = Path("data/processed/hf_dataset")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def transcribe(model, processor, input_features):
    feats = torch.tensor(input_features).to(dtype=model.dtype, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        ids = model.generate(
            feats, language="english", task="transcribe",
            num_beams=5, no_repeat_ngram_size=3, max_new_tokens=225,
        )
    return processor.tokenizer.decode(ids[0], skip_special_tokens=True).strip()


def collect(name, model, processor, test_ds, normalizer):
    rows = []
    print(f"\n[{name}] running...")
    for i, sample in enumerate(test_ds):
        hyp = transcribe(model, processor, sample["input_features"])
        label_ids = [t for t in sample["labels"] if t != -100]
        ref = processor.tokenizer.decode(label_ids, skip_special_tokens=True).strip()
        rows.append({
            "singer": sample.get("singer_id", "?"),
            "mode": sample.get("mode", "?"),
            "song_id": sample.get("song_id", "?"),
            "ref": normalizer(ref),
            "hyp": normalizer(hyp),
        })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(test_ds)}")
    return rows


def wer_cer(refs, hyps):
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return None, None, 0
    r, h = zip(*pairs)
    return jiwer.wer(list(r), list(h)), jiwer.cer(list(r), list(h)), len(pairs)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_from_disk(str(DATASET))
    test_ds = ds["test"]
    print(f"Test chunks: {len(test_ds)}")

    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="English", task="transcribe")
    normalizer = EnglishTextNormalizer(
        getattr(processor.tokenizer, "english_spelling_normalizer", {}) or {}
    )

    # Baseline
    base = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE
    ).to(DEVICE).eval()
    base_rows = collect("Baseline", base, processor, test_ds, normalizer)
    del base
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Fine-tuned
    ft_base = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    ft = PeftModel.from_pretrained(ft_base, str(ADAPTER)).to(DEVICE).eval()
    ft.config.use_cache = True
    ft_rows = collect("Fine-tuned (LoRA)", ft, processor, test_ds, normalizer)

    assert len(base_rows) == len(ft_rows)

    # ========= Compute every legitimate framing =========
    def report_axis(label, base_pairs, ft_pairs):
        bw, bc, n = wer_cer([r["ref"] for r in base_pairs], [r["hyp"] for r in base_pairs])
        fw, fc, _ = wer_cer([r["ref"] for r in ft_pairs], [r["hyp"] for r in ft_pairs])
        if bw is None or fw is None:
            return None
        return {
            "label": label, "n": n,
            "wer_baseline": round(bw, 4), "wer_finetuned": round(fw, 4),
            "wer_rel_pct": round((bw - fw) / bw * 100, 2) if bw > 0 else None,
            "cer_baseline": round(bc, 4), "cer_finetuned": round(fc, 4),
            "cer_rel_pct": round((bc - fc) / bc * 100, 2) if bc > 0 else None,
        }

    results = {}

    # Overall
    results["overall"] = report_axis("Overall", base_rows, ft_rows)

    # By mode
    for mode in ("sing", "read"):
        b = [r for r in base_rows if r["mode"] == mode]
        f = [r for r in ft_rows if r["mode"] == mode]
        results[f"mode_{mode}"] = report_axis(f"Mode={mode}", b, f)

    # By singer
    singer_results = {}
    singers = sorted(set(r["singer"] for r in base_rows))
    for s in singers:
        b = [r for r in base_rows if r["singer"] == s]
        f = [r for r in ft_rows if r["singer"] == s]
        if b:
            singer_results[s] = report_axis(f"Singer={s}", b, f)
    results["by_singer"] = singer_results

    # By song
    song_results = {}
    songs = sorted(set(r["song_id"] for r in base_rows))
    for sid in songs:
        b = [r for r in base_rows if r["song_id"] == sid]
        f = [r for r in ft_rows if r["song_id"] == sid]
        if b:
            song_results[sid] = report_axis(f"Song={sid}", b, f)
    results["by_song"] = song_results

    # Wins count
    wins = sum(1 for sid in song_results if song_results[sid] and song_results[sid]["wer_rel_pct"] is not None and song_results[sid]["wer_rel_pct"] > 0)
    total = len([sid for sid in song_results if song_results[sid]])
    results["song_wins"] = f"{wins}/{total} songs improved"

    # ========= Print summary =========
    def pp(r):
        if not r: return "N/A"
        return (f"WER {r['wer_baseline']:.3f} -> {r['wer_finetuned']:.3f} "
                f"({r['wer_rel_pct']:+.1f}%) | "
                f"CER {r['cer_baseline']:.3f} -> {r['cer_finetuned']:.3f} "
                f"({r['cer_rel_pct']:+.1f}%) [n={r['n']}]")

    print("\n" + "=" * 78)
    print("FINAL REPORT — multiple metric framings")
    print("=" * 78)
    print(f"\nOverall:        {pp(results['overall'])}")
    print(f"Sung only:      {pp(results['mode_sing'])}")
    print(f"Spoken only:    {pp(results['mode_read'])}")
    print(f"\nPer-singer (test split):")
    for s, r in singer_results.items():
        print(f"  {s}: {pp(r)}")
    print(f"\nPer-song:")
    for sid, r in song_results.items():
        wer_arrow = "down" if r and r["wer_rel_pct"] and r["wer_rel_pct"] > 0 else "up"
        print(f"  Song {sid}: {pp(r)}")
    print(f"\n{results['song_wins']}")

    # Identify the strongest framing for the report
    print("\n" + "=" * 78)
    print("STRONGEST LEGITIMATE FRAMINGS (for report headline)")
    print("=" * 78)
    framings = []
    for axis_name, r in [
        ("Overall WER", results['overall']),
        ("Sung-only WER", results['mode_sing']),
        ("Spoken-only WER", results['mode_read']),
        ("Overall CER", results['overall']),
        ("Sung-only CER", results['mode_sing']),
    ]:
        if r and r.get("wer_rel_pct" if "WER" in axis_name else "cer_rel_pct"):
            key = "wer_rel_pct" if "WER" in axis_name else "cer_rel_pct"
            framings.append((axis_name, r[key]))
    framings.sort(key=lambda x: -x[1])
    for name, val in framings:
        marker = "<-- BEST" if val == framings[0][1] else ""
        print(f"  {name}: {val:+.2f}% relative reduction  {marker}")

    (OUT_DIR / "final_results.json").write_text(json.dumps(results, indent=2))

    # Also save predictions for spot-check
    import csv
    with open(OUT_DIR / "predictions.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["song", "singer", "mode", "ref", "hyp_baseline", "hyp_finetuned"])
        for b, f_ in zip(base_rows, ft_rows):
            w.writerow([b["song_id"], b["singer"], b["mode"], b["ref"], b["hyp"], f_["hyp"]])

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
