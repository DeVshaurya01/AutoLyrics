"""Evaluation v2 — full-song WER vs canonical lyrics.

Evaluates both baseline Whisper-small and the fine-tuned LoRA adapter on
the test-split singers' full songs. Compares against canonical lyrics.
This is what we should have been measuring all along.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import torch
import jiwer
import json
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from peft import PeftModel

from autolyrics.data.nus48e import load_wav_16k_mono

RAW_DIR = Path("data/raw/nus-smc-corpus_48")
LYRICS_DIR = Path("data/interim/lyrics")
ADAPTER = Path("outputs/checkpoints/final_adapter")
OUT_DIR = Path("outputs/reports/eval_v2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
TEST_SINGERS = ["JLEE", "KENN"]
SR = 16_000


def load_lyrics() -> dict[str, str]:
    return {f.stem: f.read_text(encoding="utf-8").strip()
            for f in sorted(LYRICS_DIR.glob("[0-9][0-9].txt"))}


def chunk_audio(audio, sr=SR, chunk_s=28, overlap_s=2):
    n = len(audio)
    step = int((chunk_s - overlap_s) * sr)
    size = int(chunk_s * sr)
    pos = 0
    while pos < n:
        yield audio[pos:pos + size]
        if pos + size >= n:
            break
        pos += step


def transcribe(model, processor, audio):
    parts = []
    for piece in chunk_audio(audio):
        if len(piece) < SR:
            continue
        feats = processor.feature_extractor(
            piece, sampling_rate=SR, return_tensors="pt"
        ).input_features.to(dtype=model.dtype, device=DEVICE)
        with torch.no_grad():
            ids = model.generate(
                feats, language="english", task="transcribe",
                num_beams=5, no_repeat_ngram_size=3, max_new_tokens=225,
            )
        parts.append(processor.tokenizer.decode(ids[0], skip_special_tokens=True))
    return " ".join(parts).strip()


def eval_model(name, model, processor, lyrics):
    refs, hyps, rows = [], [], []
    for singer in TEST_SINGERS:
        for mode in ("sing", "read"):
            mode_dir = RAW_DIR / singer / mode
            if not mode_dir.is_dir():
                continue
            for wav in sorted(mode_dir.glob("*.wav")):
                song_id = wav.stem
                ref = lyrics.get(song_id, "")
                if not ref:
                    continue
                audio = load_wav_16k_mono(wav)
                hyp = transcribe(model, processor, audio)
                refs.append(ref)
                hyps.append(hyp)
                rows.append({"singer": singer, "mode": mode, "song_id": song_id,
                             "ref": ref, "hyp": hyp})

    normalizer = EnglishTextNormalizer(
        getattr(processor.tokenizer, "english_spelling_normalizer", {}) or {}
    )
    rn = [normalizer(r) for r in refs]
    hn = [normalizer(h) for h in hyps]
    paired = [(r, h) for r, h in zip(rn, hn) if r.strip()]
    r_list, h_list = zip(*paired)
    overall = jiwer.wer(list(r_list), list(h_list))

    sung_pairs = [(rows[i], rn[i], hn[i]) for i in range(len(rows)) if rows[i]["mode"] == "sing" and rn[i].strip()]
    read_pairs = [(rows[i], rn[i], hn[i]) for i in range(len(rows)) if rows[i]["mode"] == "read" and rn[i].strip()]
    sung_wer = jiwer.wer([p[1] for p in sung_pairs], [p[2] for p in sung_pairs]) if sung_pairs else None
    read_wer = jiwer.wer([p[1] for p in read_pairs], [p[2] for p in read_pairs]) if read_pairs else None

    print(f"\n[{name}]")
    print(f"  Overall WER: {overall:.4f}  on {len(paired)} test songs")
    if sung_wer is not None:
        print(f"  Sung   WER: {sung_wer:.4f}  ({len(sung_pairs)})")
    if read_wer is not None:
        print(f"  Spoken WER: {read_wer:.4f}  ({len(read_pairs)})")

    return {"overall": overall, "sung": sung_wer, "spoken": read_wer, "rows": rows}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lyrics = load_lyrics()
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-tiny", language="English", task="transcribe"
    )

    # Baseline
    base = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny", torch_dtype=DTYPE
    ).to(DEVICE).eval()
    base_res = eval_model("Baseline Whisper-small", base, processor, lyrics)
    del base
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Fine-tuned (LoRA merged)
    if ADAPTER.exists():
        ft_base = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-tiny", torch_dtype=DTYPE
        )
        ft = PeftModel.from_pretrained(ft_base, str(ADAPTER))
        ft = ft.merge_and_unload().to(DEVICE).eval()
        ft.config.use_cache = True
        ft_res = eval_model("Fine-tuned Whisper-small + LoRA", ft, processor, lyrics)
    else:
        print(f"\n[INFO] No adapter at {ADAPTER}, skipping fine-tuned eval.")
        ft_res = None

    # Save report
    out = {
        "baseline": {k: v for k, v in base_res.items() if k != "rows"},
        "finetuned": {k: v for k, v in ft_res.items() if k != "rows"} if ft_res else None,
    }
    if ft_res:
        rel = (base_res["overall"] - ft_res["overall"]) / base_res["overall"] * 100
        out["relative_wer_reduction_pct"] = round(rel, 2)
        print(f"\n[RESULT] Relative WER reduction: {rel:.2f}%")

    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {OUT_DIR/'results.json'}")


if __name__ == "__main__":
    main()
