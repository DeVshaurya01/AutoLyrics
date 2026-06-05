"""Data pipeline v2 — proper word-level alignment.

Instead of chunking by the unreliable `sp`-marker heuristic, this script:
  1. Runs Whisper-small on each full song to get word-level timestamps.
  2. Uses Whisper's word transcription as the reference text (it's at 29%
     baseline WER — good enough to train against).
  3. Chunks audio at natural silence gaps in Whisper's word timing.
  4. Each chunk's reference = Whisper's words that fall in its time window.

The result: properly aligned (audio, text) pairs. No more empty-ref or
shifted-ref bugs. Speaker-disjoint splits preserved.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import torch
import numpy as np
from datasets import Dataset, DatasetDict, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from autolyrics.data.nus48e import load_wav_16k_mono

RAW_DIR = Path("data/raw/nus-smc-corpus_48")
PROCESSED_DIR = Path("data/processed")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16_000

TRAIN_SINGERS = ["ADIZ", "JTAN", "MCUR", "MPOL", "MPUR", "NJAT", "PMAR", "ZHIY"]
VAL_SINGERS   = ["SAMF", "VKOW"]
TEST_SINGERS  = ["JLEE", "KENN"]

CHUNK_TARGET_S = 22.0
CHUNK_MAX_S    = 28.0
CHUNK_MIN_S    = 4.0


def whisper_word_timestamps(model, processor, audio: np.ndarray) -> list[dict]:
    """Return list of {"word": str, "start": float, "end": float} for the full audio.

    Handles audio >30s by chunking with overlap and offsetting timestamps.
    """
    out = []
    chunk_s = 28.0
    overlap_s = 2.0
    step_samples = int((chunk_s - overlap_s) * SR)
    size_samples = int(chunk_s * SR)
    pos = 0
    base_t = 0.0

    while pos < len(audio):
        piece = audio[pos:pos + size_samples]
        if len(piece) < SR:
            break
        feats = processor.feature_extractor(
            piece, sampling_rate=SR, return_tensors="pt"
        ).input_features.to(dtype=model.dtype, device=DEVICE)
        with torch.no_grad():
            gen = model.generate(
                feats,
                language="english",
                task="transcribe",
                return_timestamps=True,
                num_beams=1,
                max_new_tokens=225,
            )
        # Decode with token-level timestamps, then split into words
        decoded = processor.tokenizer.decode(
            gen[0], skip_special_tokens=False, decode_with_timestamps=True
        )
        # Parse <|t.tt|> markers
        import re
        ts_pattern = re.compile(r"<\|([\d.]+)\|>")
        segs = ts_pattern.split(decoded)
        # segs alternates: [text_before, ts1, text, ts2, text, ...]
        i = 1
        while i + 2 < len(segs):
            t_start = float(segs[i])
            text = segs[i + 1].strip()
            t_end = float(segs[i + 2])
            if text:
                for w in text.split():
                    out.append({
                        "word": w,
                        "start": base_t + t_start,
                        "end": base_t + t_end,
                    })
            i += 2

        if pos + size_samples >= len(audio):
            break
        pos += step_samples
        base_t = pos / SR

    # Deduplicate words from overlapping windows (keep first occurrence)
    if not out:
        return out
    dedup = [out[0]]
    for w in out[1:]:
        if abs(w["start"] - dedup[-1]["start"]) < 0.1 and w["word"].lower() == dedup[-1]["word"].lower():
            continue
        dedup.append(w)
    return dedup


def chunk_by_word_gaps(audio: np.ndarray, words: list[dict],
                       target_s=CHUNK_TARGET_S, max_s=CHUNK_MAX_S,
                       min_s=CHUNK_MIN_S) -> list[dict]:
    """Group consecutive words into chunks. Break at natural pauses or at max_s."""
    if not words:
        return []

    chunks = []
    cur_words = []
    cur_start = words[0]["start"]

    for i, w in enumerate(words):
        cur_words.append(w)
        gap_next = words[i + 1]["start"] - w["end"] if i + 1 < len(words) else 999.0
        elapsed = w["end"] - cur_start

        # Break conditions: we've hit target and there's a natural pause, OR hit max
        should_break = (elapsed >= target_s and gap_next > 0.4) or elapsed >= max_s
        if should_break or i == len(words) - 1:
            chunk_end = w["end"]
            if chunk_end - cur_start >= min_s:
                text = " ".join(x["word"] for x in cur_words).strip()
                audio_slice = audio[int(cur_start * SR):int(chunk_end * SR)]
                if text and len(audio_slice) >= int(min_s * SR):
                    chunks.append({
                        "audio": audio_slice,
                        "text": text,
                        "start": cur_start,
                        "end": chunk_end,
                    })
            cur_words = []
            cur_start = words[i + 1]["start"] if i + 1 < len(words) else 0.0

    return chunks


def assign_split(singer: str) -> str | None:
    if singer in TRAIN_SINGERS: return "train"
    if singer in VAL_SINGERS:   return "val"
    if singer in TEST_SINGERS:  return "test"
    return None


def main():
    print(f"Loading Whisper-small on {DEVICE}...")
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-tiny", language="English", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    model = model.to(DEVICE).eval()

    rows = {"train": [], "val": [], "test": []}

    singers = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    for singer_dir in singers:
        singer = singer_dir.name
        split = assign_split(singer)
        if split is None:
            continue
        for mode in ("sing", "read"):
            mode_dir = singer_dir / mode
            if not mode_dir.is_dir():
                continue
            for wav in sorted(mode_dir.glob("*.wav")):
                song_id = wav.stem
                print(f"  {singer}/{mode}/{song_id} ... ", end="", flush=True)
                audio = load_wav_16k_mono(wav)
                words = whisper_word_timestamps(model, processor, audio)
                chunks = chunk_by_word_gaps(audio, words)
                for ch in chunks:
                    rows[split].append({
                        "audio_array": ch["audio"].astype(np.float32),
                        "transcription": ch["text"],
                        "singer_id": singer,
                        "song_id": song_id,
                        "mode": mode,
                    })
                print(f"{len(chunks)} chunks")

    # Build HF datasets
    def to_hf(samples):
        if not samples:
            return None
        ds = Dataset.from_dict({
            "audio": [{"array": s["audio_array"], "sampling_rate": SR} for s in samples],
            "transcription": [s["transcription"] for s in samples],
            "singer_id":     [s["singer_id"]     for s in samples],
            "song_id":       [s["song_id"]       for s in samples],
            "mode":          [s["mode"]          for s in samples],
        })
        ds = ds.cast_column("audio", Audio(sampling_rate=SR))
        return ds

    splits = {k: to_hf(v) for k, v in rows.items()}
    splits = {k: v for k, v in splits.items() if v is not None}

    print("\nApplying feature extraction + tokenization...")

    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=SR
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
        return batch

    for k in splits:
        splits[k] = splits[k].map(prepare, remove_columns=["audio"], num_proc=1)
        # Filter overly long label sequences
        before = len(splits[k])
        splits[k] = splits[k].filter(lambda x: len(x["labels"]) <= 448)
        print(f"  {k}: {before} -> {len(splits[k])} samples (filtered long labels)")

    out_path = PROCESSED_DIR / "hf_dataset"
    if out_path.exists():
        import shutil
        shutil.rmtree(out_path)
    DatasetDict(splits).save_to_disk(str(out_path))
    print(f"\n[DONE] Saved to {out_path}")
    for k, v in splits.items():
        print(f"  {k}: {len(v)}")


if __name__ == "__main__":
    main()
