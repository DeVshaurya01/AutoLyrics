"""Entry point: raw NUS-48E → data/processed/hf_dataset + Arrow shards."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from pathlib import Path

from autolyrics.utils.paths import DATA_DIR
from autolyrics.data.preprocess import (
    build_manifests,
    save_manifests,
    save_audio_chunks,
    build_hf_dataset,
)
from autolyrics.models.whisper_lora import build_processor


def main():
    parser = argparse.ArgumentParser(description="Prepare NUS-48E for training")
    parser.add_argument(
        "--raw_dir",
        default=str(DATA_DIR / "raw" / "nus-smc-corpus_48"),
        help="Path to extracted NUS-48E corpus root",
    )
    parser.add_argument(
        "--lyrics_dir",
        default=str(DATA_DIR / "interim" / "lyrics"),
        help="Path to canonical lyrics files (01.txt … 20.txt)",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DATA_DIR / "processed"),
        help="Root output directory for chunks + HF dataset",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    lyrics_dir = Path(args.lyrics_dir)
    output_dir = Path(args.output_dir)

    if not raw_dir.exists():
        print(f"[ERROR] Raw corpus not found at {raw_dir}")
        print("  Download NUS-48E and place it there. See build.md §10.")
        raise SystemExit(1)

    print("=" * 60)
    print("AutoLyrics — Data Preparation Pipeline")
    print("=" * 60)

    print("\n[1/4] Building chunk manifests from NUS-48E corpus...")
    manifests = build_manifests(raw_dir, lyrics_dir)
    total = sum(len(v) for v in manifests.values())
    print(f"      Total chunks: {total}")
    for split, samples in manifests.items():
        print(f"      {split}: {len(samples)} chunks")

    if total == 0:
        print("[ERROR] No chunks produced. Check raw data structure and lyrics files.")
        raise SystemExit(1)

    print("\n[2/4] Saving manifest JSONs...")
    manifest_dir = DATA_DIR / "interim" / "manifests"
    save_manifests(manifests, manifest_dir)

    print("\n[3/4] Writing chunk WAV files...")
    manifests_with_paths = save_audio_chunks(manifests, output_dir)

    print("\n[4/4] Building HuggingFace DatasetDict with mel features + labels...")
    processor = build_processor()
    hf_ds = build_hf_dataset(manifests_with_paths, processor)

    hf_path = output_dir / "hf_dataset"
    hf_ds.save_to_disk(str(hf_path))
    print(f"\n[DONE] HF dataset saved to {hf_path}")
    for split in hf_ds:
        print(f"       {split}: {len(hf_ds[split])} samples")


if __name__ == "__main__":
    main()
