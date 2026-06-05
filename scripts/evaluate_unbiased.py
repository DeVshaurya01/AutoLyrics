"""Unbiased test-set evaluation.

prepare_data_v3 used whisper itself to align chunks and dropped songs with
<40% alignment coverage. That pre-selects chunks whisper can already
transcribe, deflating baseline WER and erasing improvement headroom.

This script rebuilds the test set from scratch using ONLY NUS-48E's own
phone labels (Audacity .txt files) — independent of any whisper output.
Word/timestamp alignment uses the `sp` markers in the phone tier
(each sp ≈ word boundary, per Duan et al. APSIPA 2013).

Singers: JLEE, KENN (sung only). Audio is chunked at sil/sp boundaries
to ~22s, capped at 28s. Each chunk's reference text is the canonical
lyrics slice assigned by sp counting.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from pathlib import Path

import torch
import jiwer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from peft import PeftModel

from autolyrics.data.nus48e import (
    load_wav_16k_mono, parse_nus_label, build_lyrics_map, chunk_by_silence,
)

RAW_DIR = Path("data/raw/nus-smc-corpus_48")
LYRICS_DIR = Path("data/interim/lyrics")
ADAPTER = Path("outputs/checkpoints/final_adapter")
OUT_DIR = Path("outputs/reports/eval_unbiased")
MODEL_ID = "openai/whisper-small"
TEST_SINGERS = ["JLEE", "KENN"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def build_test_chunks():
    """Walk JLEE/KENN sung audio and emit phone-label-aligned chunks."""
    lyrics_map = build_lyrics_map(LYRICS_DIR)
    chunks_out = []
    for singer in TEST_SINGERS:
        sing_dir = RAW_DIR / singer / "sing"
        if not sing_dir.is_dir():
            continue
        for wav in sorted(sing_dir.glob("*.wav")):
            song_id = wav.stem
            label = wav.with_suffix(".txt")
            if not label.exists() or song_id not in lyrics_map:
                continue
            audio = load_wav_16k_mono(wav)
            segs = parse_nus_label(label)
            song_chunks = chunk_by_silence(audio, segs, lyrics_map[song_id])
            for ch in song_chunks:
                if ch["text"].strip():
                    chunks_out.append({
                        "audio": ch["audio"],
                        "text": ch["text"],
                        "singer": singer,
                        "song_id": song_id,
                    })
    return chunks_out


def transcribe(model, processor, audio_np):
    feats = processor.feature_extractor(
        audio_np, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(dtype=model.dtype, device=DEVICE)
    with torch.no_grad():
        ids = model.generate(
            feats, language="english", task="transcribe",
            num_beams=5, no_repeat_ngram_size=3, max_new_tokens=225,
        )
    return processor.tokenizer.decode(ids[0], skip_special_tokens=True).strip()


def eval_model(name, model, processor, chunks, normalizer):
    refs, hyps = [], []
    for i, c in enumerate(chunks):
        hyps.append(transcribe(model, processor, c["audio"]))
        refs.append(c["text"])
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(chunks)}")
    rn = [normalizer(r) for r in refs]
    hn = [normalizer(h) for h in hyps]
    paired = [(r, h) for r, h in zip(rn, hn) if r.strip()]
    r_list, h_list = zip(*paired)
    wer = jiwer.wer(list(r_list), list(h_list))
    cer = jiwer.cer(list(r_list), list(h_list))
    print(f"\n[{name}] WER={wer:.4f}  CER={cer:.4f}  on {len(paired)} chunks")
    return {"wer": wer, "cer": cer, "n": len(paired), "refs": refs, "hyps": hyps}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building unbiased test chunks from NUS phone labels...")
    chunks = build_test_chunks()
    print(f"  {len(chunks)} chunks  (singers: {TEST_SINGERS}, sung only)")

    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="English", task="transcribe")
    normalizer = EnglishTextNormalizer(
        getattr(processor.tokenizer, "english_spelling_normalizer", {}) or {}
    )

    print("\nLoading baseline whisper-small...")
    base = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE).eval()
    base_res = eval_model("Baseline", base, processor, chunks, normalizer)
    del base
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    ft_res = None
    if ADAPTER.exists():
        print("\nLoading fine-tuned LoRA...")
        ft_base = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
        ft = PeftModel.from_pretrained(ft_base, str(ADAPTER)).to(DEVICE).eval()
        ft.config.use_cache = True
        ft_res = eval_model("Fine-tuned (LoRA)", ft, processor, chunks, normalizer)

    if ft_res:
        rel = (base_res["wer"] - ft_res["wer"]) / base_res["wer"] * 100
        print(f"\n[RESULT] Relative WER reduction: {rel:.2f}%   (target ≥15%)")
        out = {
            "n_chunks": len(chunks),
            "baseline_wer": base_res["wer"],
            "baseline_cer": base_res["cer"],
            "finetuned_wer": ft_res["wer"],
            "finetuned_cer": ft_res["cer"],
            "relative_wer_reduction_pct": rel,
        }
        (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2, default=str))
        # Save preds for inspection
        rows = [{"ref": r, "hyp_base": hb, "hyp_ft": hf}
                for r, hb, hf in zip(base_res["refs"], base_res["hyps"], ft_res["hyps"])]
        (OUT_DIR / "predictions.json").write_text(json.dumps(rows, indent=2))
        print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
