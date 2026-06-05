"""Spec-faithful prep: NUS-48E sung audio only, chunked via phone labels.

No Jamendo, no whisper-aligned chunks, no augmentation. The same
chunk_by_silence used in evaluate_unbiased.py — so train/val/test
distributions match exactly.

Singers (build.md):
  train: ADIZ, JTAN, MCUR, MPOL, MPUR, NJAT, PMAR, ZHIY
  val:   SAMF, VKOW
  test:  JLEE, KENN
Mode: sing only.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import shutil
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, Audio
from transformers import WhisperProcessor

from autolyrics.data.nus48e import (
    load_wav_16k_mono, parse_nus_label, build_lyrics_map, chunk_by_silence,
)

RAW_DIR = Path("data/raw/nus-smc-corpus_48")
LYRICS_DIR = Path("data/interim/lyrics")
PROCESSED = Path("data/processed/hf_dataset")
MODEL_ID = "openai/whisper-small"
SR = 16_000

TRAIN_SINGERS = ["ADIZ", "JTAN", "MCUR", "MPOL", "MPUR", "NJAT", "PMAR", "ZHIY"]
VAL_SINGERS = ["SAMF", "VKOW"]
TEST_SINGERS = ["JLEE", "KENN"]


def split_of(singer):
    if singer in TRAIN_SINGERS: return "train"
    if singer in VAL_SINGERS: return "val"
    if singer in TEST_SINGERS: return "test"
    return None


def main():
    lyrics_map = build_lyrics_map(LYRICS_DIR)
    rows = {"train": [], "val": [], "test": []}

    for singer_dir in sorted(d for d in RAW_DIR.iterdir() if d.is_dir()):
        singer = singer_dir.name
        split = split_of(singer)
        if split is None:
            continue
        sing_dir = singer_dir / "sing"
        if not sing_dir.is_dir():
            continue
        for wav in sorted(sing_dir.glob("*.wav")):
            song_id = wav.stem
            label = wav.with_suffix(".txt")
            if not label.exists() or song_id not in lyrics_map:
                continue
            audio = load_wav_16k_mono(wav)
            segs = parse_nus_label(label)
            chunks = chunk_by_silence(audio, segs, lyrics_map[song_id])
            kept = 0
            for ch in chunks:
                if ch["text"].strip():
                    rows[split].append({
                        "audio_array": ch["audio"].astype(np.float32),
                        "transcription": ch["text"],
                        "singer_id": singer,
                        "song_id": song_id,
                        "mode": "sing",
                    })
                    kept += 1
            print(f"  {singer}/{song_id}: {kept} chunks")

    print("\nSplit sizes:")
    for k, v in rows.items():
        print(f"  {k}: {len(v)}")

    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="English", task="transcribe")

    def to_hf(samples):
        if not samples:
            return None
        ds = Dataset.from_dict({
            "audio": [{"array": s["audio_array"], "sampling_rate": SR} for s in samples],
            "transcription": [s["transcription"] for s in samples],
            "singer_id": [s["singer_id"] for s in samples],
            "song_id": [s["song_id"] for s in samples],
            "mode": [s["mode"] for s in samples],
        })
        return ds.cast_column("audio", Audio(sampling_rate=SR))

    splits = {k: to_hf(v) for k, v in rows.items() if v}

    def prepare(batch):
        a = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            a["array"], sampling_rate=SR
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
        return batch

    for k in splits:
        splits[k] = splits[k].map(prepare, remove_columns=["audio"], num_proc=1)
        before = len(splits[k])
        splits[k] = splits[k].filter(lambda x: len(x["labels"]) <= 448)
        print(f"  {k}: {before} -> {len(splits[k])} after long-label filter")

    if PROCESSED.exists():
        shutil.rmtree(PROCESSED)
    DatasetDict(splits).save_to_disk(str(PROCESSED))
    print(f"\n[DONE] Saved to {PROCESSED}")


if __name__ == "__main__":
    main()
