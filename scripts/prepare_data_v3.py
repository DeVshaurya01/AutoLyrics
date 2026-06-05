"""Data pipeline v3 — canonical lyrics aligned to audio via Whisper timestamps.

Pipeline:
  1. Whisper-small transcribes each full song with word-level timestamps.
  2. Sequence-align Whisper's hypothesis words to canonical lyric words
     (difflib.SequenceMatcher on normalized tokens).
  3. Inherit timestamps from matched Whisper words; interpolate for unmatched
     canonical words.
  4. Chunk the canonical word sequence into 20-25s segments using interpolated
     timestamps. Each chunk's audio = time slice, text = canonical words.
  5. Drop songs/segments with poor alignment coverage (<40% matched words).

This produces properly-aligned (audio, canonical_text) pairs — the legitimate
training target for adapting Whisper to NUS-48E.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path

import torch
import torchaudio
import numpy as np
from datasets import Dataset, DatasetDict, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from autolyrics.data.nus48e import load_wav_16k_mono

RAW_DIR = Path("data/raw/nus-smc-corpus_48")
LYRICS_DIR = Path("data/interim/lyrics")
PROCESSED_DIR = Path("data/processed")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16_000

TRAIN_SINGERS = ["ADIZ", "JTAN", "MCUR", "MPOL", "MPUR", "NJAT", "PMAR", "ZHIY"]
VAL_SINGERS   = ["SAMF", "VKOW"]
TEST_SINGERS  = ["JLEE", "KENN"]

CHUNK_TARGET_S = 22.0
CHUNK_MAX_S    = 28.0
CHUNK_MIN_S    = 4.0
CHUNK_PAD_S    = 0.15   # buffer on both ends so we don't clip word edges
MIN_COVERAGE   = 0.40   # drop songs where <40% of canonical words got matched


def assign_split(singer):
    if singer in TRAIN_SINGERS: return "train"
    if singer in VAL_SINGERS:   return "val"
    if singer in TEST_SINGERS:  return "test"
    return None


def normalize_token(t):
    """Lowercase, strip non-letters/apostrophes. Used for alignment matching only."""
    return re.sub(r"[^a-z']", "", t.lower())


def whisper_word_timestamps(model, processor, audio):
    """Return [{"word": str, "start": float, "end": float}, ...] for full audio."""
    out = []
    chunk_s, overlap_s = 28.0, 2.0
    step = int((chunk_s - overlap_s) * SR)
    size = int(chunk_s * SR)
    pos = 0
    base_t = 0.0

    ts_pattern = re.compile(r"<\|([\d.]+)\|>")
    while pos < len(audio):
        piece = audio[pos:pos + size]
        if len(piece) < SR:
            break
        feats = processor.feature_extractor(
            piece, sampling_rate=SR, return_tensors="pt"
        ).input_features.to(dtype=model.dtype, device=DEVICE)
        with torch.no_grad():
            gen = model.generate(
                feats, language="english", task="transcribe",
                return_timestamps=True, num_beams=1, max_new_tokens=225,
            )
        decoded = processor.tokenizer.decode(
            gen[0], skip_special_tokens=False, decode_with_timestamps=True
        )
        segs = ts_pattern.split(decoded)
        i = 1
        while i + 2 < len(segs):
            t_start = float(segs[i])
            text = segs[i + 1].strip()
            t_end = float(segs[i + 2])
            if text:
                # Distribute words evenly across the segment time range
                ws = text.split()
                if ws:
                    span = max(t_end - t_start, 0.01)
                    per = span / len(ws)
                    for k, w in enumerate(ws):
                        out.append({
                            "word": w,
                            "start": base_t + t_start + k * per,
                            "end": base_t + t_start + (k + 1) * per,
                        })
            i += 2
        if pos + size >= len(audio):
            break
        pos += step
        base_t = pos / SR

    # Dedup overlapping-window duplicates
    if not out:
        return out
    dedup = [out[0]]
    for w in out[1:]:
        if (abs(w["start"] - dedup[-1]["start"]) < 0.15 and
                normalize_token(w["word"]) == normalize_token(dedup[-1]["word"])):
            continue
        dedup.append(w)
    return dedup


def align_canon_to_whisper(canon_words, whisper_words):
    """Return list of (canon_start_s, canon_end_s) per canonical word.

    Unmatched canonical words get timestamps interpolated from their matched
    neighbors. Returns also the coverage ratio (matched / total).
    """
    canon_norm = [normalize_token(w) for w in canon_words]
    whisp_norm = [normalize_token(w["word"]) for w in whisper_words]
    matcher = SequenceMatcher(a=canon_norm, b=whisp_norm, autojunk=False)

    matched_idx = [None] * len(canon_words)  # canon_i -> whisper_j
    for i, j, n in matcher.get_matching_blocks():
        for k in range(n):
            matched_idx[i + k] = j + k

    coverage = sum(1 for x in matched_idx if x is not None) / max(len(canon_words), 1)

    # Build (start, end) for each canon word; interpolate gaps linearly.
    times = [None] * len(canon_words)
    for ci, wj in enumerate(matched_idx):
        if wj is not None:
            times[ci] = (whisper_words[wj]["start"], whisper_words[wj]["end"])

    # Interpolate unmatched runs between two anchors
    def find_next_anchor(start):
        for k in range(start, len(times)):
            if times[k] is not None:
                return k
        return None

    last_anchor = -1
    for i in range(len(times)):
        if times[i] is not None:
            # fill range (last_anchor, i)
            if last_anchor != i - 1:
                left_end = times[last_anchor][1] if last_anchor >= 0 else times[i][0]
                right_start = times[i][0]
                gap_count = i - last_anchor - 1
                if gap_count > 0:
                    step = (right_start - left_end) / (gap_count + 1)
                    for k in range(gap_count):
                        t = left_end + step * (k + 1)
                        times[last_anchor + 1 + k] = (t, t + max(step, 0.1))
            last_anchor = i

    # Tail (after last anchor)
    if last_anchor >= 0 and last_anchor < len(times) - 1:
        last_t = times[last_anchor][1]
        for k in range(last_anchor + 1, len(times)):
            times[k] = (last_t, last_t + 0.2)
            last_t += 0.2

    # Head (before first anchor) — if first canon words have no match
    first_anchor = next((i for i, t in enumerate(times) if t is not None), None)
    if first_anchor and first_anchor > 0:
        first_t = times[first_anchor][0]
        for k in range(first_anchor):
            times[k] = (max(first_t - 0.2 * (first_anchor - k), 0.0), first_t)

    return times, coverage


def chunk_canonical(canon_words, times, audio,
                    target_s=CHUNK_TARGET_S, max_s=CHUNK_MAX_S, min_s=CHUNK_MIN_S):
    """Group canonical words into <=max_s chunks; use interpolated timestamps."""
    if not canon_words or not times or times[0] is None:
        return []

    chunks = []
    cur = []
    cur_start = times[0][0]

    for i, w in enumerate(canon_words):
        if times[i] is None:
            continue
        cur.append((w, times[i]))
        elapsed = times[i][1] - cur_start
        gap_next = (times[i + 1][0] - times[i][1]) if (i + 1 < len(times) and times[i + 1] is not None) else 999.0

        should_break = (elapsed >= target_s and gap_next > 0.3) or elapsed >= max_s
        if should_break or i == len(canon_words) - 1:
            chunk_end = cur[-1][1][1]
            if chunk_end - cur_start >= min_s:
                # Pad both ends so forced-alignment slop doesn't clip word edges
                a0 = max(int((cur_start - CHUNK_PAD_S) * SR), 0)
                a1 = min(int((chunk_end + CHUNK_PAD_S) * SR), len(audio))
                audio_slice = audio[a0:a1]
                text = " ".join(x[0] for x in cur).strip()
                if text and len(audio_slice) >= int(min_s * SR):
                    chunks.append({
                        "audio": audio_slice,
                        "text": text,
                        "start": cur_start,
                        "end": chunk_end,
                    })
            cur = []
            # next chunk starts at next valid timestamp
            for k in range(i + 1, len(times)):
                if times[k] is not None:
                    cur_start = times[k][0]
                    break
    return chunks


def main():
    print(f"Loading Whisper-small on {DEVICE}...")
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="English", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model = model.to(DEVICE).eval()

    # Load canonical lyrics
    lyrics_map = {
        f.stem: f.read_text(encoding="utf-8").strip().split()
        for f in sorted(LYRICS_DIR.glob("[0-9][0-9].txt"))
    }
    print(f"Loaded canonical lyrics for {len(lyrics_map)} songs.\n")

    rows = {"train": [], "val": [], "test": []}
    skipped = []

    for singer_dir in sorted(d for d in RAW_DIR.iterdir() if d.is_dir()):
        singer = singer_dir.name
        split = assign_split(singer)
        if split is None:
            continue
        for mode in ("sing", "read"):
            if split == "test" and mode == "read":
                continue
            mode_dir = singer_dir / mode
            if not mode_dir.is_dir():
                continue
            # Use both sung and spoken in training. Spoken acts as
            # regularization (prevents over-specializing to singing
            # acoustics) and gives the model more aligned data to learn from.
            for wav in sorted(mode_dir.glob("*.wav")):
                song_id = wav.stem
                canon = lyrics_map.get(song_id, [])
                if not canon:
                    continue

                audio = load_wav_16k_mono(wav)
                whisp = whisper_word_timestamps(model, processor, audio)
                if not whisp:
                    skipped.append((singer, mode, song_id, "no_whisper_output"))
                    print(f"  {singer}/{mode}/{song_id}  SKIPPED (no whisper output)")
                    continue

                times, coverage = align_canon_to_whisper(canon, whisp)
                if coverage < MIN_COVERAGE:
                    skipped.append((singer, mode, song_id, f"low_coverage_{coverage:.2f}"))
                    print(f"  {singer}/{mode}/{song_id}  SKIPPED (coverage={coverage:.2f})")
                    continue

                chunks = chunk_canonical(canon, times, audio)
                for ch in chunks:
                    rows[split].append({
                        "audio_array": ch["audio"].astype(np.float32),
                        "transcription": ch["text"],
                        "singer_id": singer,
                        "song_id": song_id,
                        "mode": mode,
                    })
                print(f"  {singer}/{mode}/{song_id}  cov={coverage:.2f}  chunks={len(chunks)}")

    print(f"\nSkipped {len(skipped)} recordings due to alignment issues:")
    for s in skipped[:10]:
        print(f"  {s}")
    if len(skipped) > 10:
        print(f"  ... and {len(skipped) - 10} more")

    # ====== Add JamendoLyrics chunks to training data ======
    jamendo_manifest = Path("data/raw/jamendo/_processed/manifest.json")
    if jamendo_manifest.exists():
        import json as _json
        items = _json.loads(jamendo_manifest.read_text())
        print(f"\n[INFO] Adding {len(items)} Jamendo chunks to TRAINING split...")
        for it in items:
            audio_arr = np.load(it["audio_path"])
            rows["train"].append({
                "audio_array": audio_arr.astype(np.float32),
                "transcription": it["text"],
                "singer_id": f"jamendo_{it['song']}",
                "song_id":   f"j_{it['song'][:20]}",
                "mode": "sing",
            })
        print(f"[INFO] Train pool now: {len(rows['train'])} chunks "
              f"(includes Jamendo augmentation)")
    else:
        print(f"\n[INFO] No Jamendo data at {jamendo_manifest}. "
              f"Run scripts/fetch_jamendo.py to add.")

    # ====== Time-stretch augmentation on TRAINING chunks (±10%) ======
    print("\n[INFO] Applying ±10% time-stretch augmentation to training chunks...")
    import random as _random
    _random.seed(42)
    augmented = []
    for r in rows["train"]:
        # Original
        augmented.append(r)
        # Two augmented copies per original: slow and fast
        for rate in (0.9, 1.1):
            wav_t = torch.from_numpy(r["audio_array"]).unsqueeze(0)
            try:
                stretched = torchaudio.functional.speed(wav_t, SR, rate)[0]
                augmented.append({
                    **r,
                    "audio_array": stretched.squeeze(0).numpy().astype("float32"),
                })
            except Exception:
                pass
    rows["train"] = augmented
    print(f"[INFO] Train pool after augmentation: {len(rows['train'])} chunks")

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
        return ds.cast_column("audio", Audio(sampling_rate=SR))

    splits = {k: to_hf(v) for k, v in rows.items() if v}

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
        before = len(splits[k])
        splits[k] = splits[k].filter(lambda x: len(x["labels"]) <= 448)
        print(f"  {k}: {before} -> {len(splits[k])} samples (filtered long labels)")

    out_path = PROCESSED_DIR / "hf_dataset"
    if out_path.exists():
        shutil.rmtree(out_path)
    DatasetDict(splits).save_to_disk(str(out_path))
    print(f"\n[DONE] Saved to {out_path}")
    for k, v in splits.items():
        print(f"  {k}: {len(v)} samples")


if __name__ == "__main__":
    main()
