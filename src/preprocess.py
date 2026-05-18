"""
AutoLyrics — Data Download & Preprocessing Pipeline
Downloads NUS-48E dataset and prepares it for Whisper fine-tuning.
"""

import os
import json
import argparse
import zipfile
import shutil
from pathlib import Path

import torch
import torchaudio
import librosa
import numpy as np
from transformers import WhisperProcessor

# ──────────────────────────────────────────────────────────────────────
# NUS-48E Dataset Info
# ──────────────────────────────────────────────────────────────────────
# NUS-48E contains 48 English songs sung by 12 singers (6 male, 6 female).
# Each song has: sung audio (.wav) + ground-truth lyrics (.txt)
# Official site: https://smcnus.comp.nus.edu.sg/nus-48e-sung-and-spoken-lyrics-corpus/
#
# Since direct automated download may not always work, this script supports:
#   1. Automatic download attempt from known mirrors
#   2. Manual placement: download the zip yourself and point --data_dir to it
# ──────────────────────────────────────────────────────────────────────

NUS48E_HF_REPO = "Jayeshbhaal/NUS-48E"  # Community HuggingFace mirror


def download_nus48e(data_dir: str, use_hf: bool = True) -> Path:
    """
    Download NUS-48E dataset.

    Args:
        data_dir: Root directory to store raw data.
        use_hf: If True, try HuggingFace datasets first.

    Returns:
        Path to the raw dataset directory.
    """
    raw_dir = Path(data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    if any(raw_dir.glob("**/*.wav")):
        print(f"[INFO] Found existing audio files in {raw_dir}, skipping download.")
        return raw_dir

    if use_hf:
        print("[INFO] Attempting download from HuggingFace...")
        try:
            from datasets import load_dataset

            ds = load_dataset(NUS48E_HF_REPO, trust_remote_code=True)
            # Save to disk for our pipeline
            hf_cache = raw_dir / "hf_cache"
            ds.save_to_disk(str(hf_cache))
            print(f"[INFO] Dataset saved to {hf_cache}")
            return hf_cache
        except Exception as e:
            print(f"[WARN] HuggingFace download failed: {e}")
            print("[INFO] Please download NUS-48E manually:")
            print("  1. Visit https://smcnus.comp.nus.edu.sg/nus-48e-sung-and-spoken-lyrics-corpus/")
            print(f"  2. Extract the zip contents into: {raw_dir}")
            print("  3. Re-run this script.")
            raise

    return raw_dir


def find_audio_lyric_pairs(raw_dir: Path) -> list[dict]:
    """
    Scan the raw dataset directory and pair audio files with their lyrics.

    Returns:
        List of dicts with keys: audio_path, lyrics, singer, song_name
    """
    pairs = []

    # Strategy 1: HuggingFace datasets format (Arrow files)
    hf_cache = raw_dir / "hf_cache"
    if hf_cache.exists():
        from datasets import load_from_disk

        ds = load_from_disk(str(hf_cache))
        # Iterate over all splits
        for split_name in ds:
            split = ds[split_name]
            for i, sample in enumerate(split):
                # HF datasets typically store audio as dict with 'array' and 'sampling_rate'
                audio_data = sample.get("audio", None)
                lyrics = sample.get("lyrics", sample.get("text", sample.get("transcription", "")))

                if audio_data is not None and lyrics:
                    pairs.append({
                        "index": i,
                        "split": split_name,
                        "audio": audio_data,
                        "lyrics": lyrics.strip(),
                        "singer": sample.get("singer", "unknown"),
                        "song_name": sample.get("song", f"sample_{i}"),
                    })
        return pairs

    # Strategy 2: Standard directory structure (wav + txt files)
    for wav_path in sorted(raw_dir.rglob("*.wav")):
        # Look for matching lyrics file
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            # Try alternative naming: song_lyrics.txt
            txt_path = wav_path.parent / (wav_path.stem + "_lyrics.txt")
        if not txt_path.exists():
            # Try looking in a lyrics subdirectory
            txt_path = wav_path.parent.parent / "lyrics" / (wav_path.stem + ".txt")

        if txt_path.exists():
            lyrics = txt_path.read_text(encoding="utf-8").strip()
            parts = wav_path.stem.split("_")
            pairs.append({
                "audio_path": str(wav_path),
                "lyrics": lyrics,
                "singer": parts[0] if len(parts) > 1 else "unknown",
                "song_name": wav_path.stem,
            })
        else:
            print(f"[WARN] No lyrics found for {wav_path.name}, skipping.")

    print(f"[INFO] Found {len(pairs)} audio-lyric pairs.")
    return pairs


def preprocess_audio(audio_path: str = None, audio_data: dict = None,
                     target_sr: int = 16000) -> np.ndarray:
    """
    Load and resample audio to 16kHz mono (Whisper's expected input).

    Args:
        audio_path: Path to audio file.
        audio_data: HuggingFace audio dict with 'array' and 'sampling_rate'.
        target_sr: Target sample rate.

    Returns:
        Audio as numpy array at target sample rate.
    """
    if audio_data is not None:
        # HuggingFace format
        waveform = np.array(audio_data["array"], dtype=np.float32)
        sr = audio_data["sampling_rate"]
        if sr != target_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        return waveform

    # File-based loading
    waveform, sr = torchaudio.load(audio_path)
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform.squeeze().numpy()


def prepare_dataset(pairs: list[dict], processor: WhisperProcessor,
                    output_dir: str, split_ratios: tuple = (0.8, 0.1, 0.1)):
    """
    Process all audio-lyric pairs and save as train/val/test splits.

    Each sample is saved as a dict with:
        - input_features: mel spectrogram (from Whisper's feature extractor)
        - labels: tokenized lyrics (from Whisper's tokenizer)
        - metadata: singer, song name, etc.
    """
    output_path = Path(output_dir)

    # Shuffle deterministically
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(pairs))

    n_train = int(len(pairs) * split_ratios[0])
    n_val = int(len(pairs) * split_ratios[1])

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }

    for split_name, split_indices in splits.items():
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        samples = []
        for idx in split_indices:
            pair = pairs[idx]

            # Load and preprocess audio
            if "audio_path" in pair:
                waveform = preprocess_audio(audio_path=pair["audio_path"])
            elif "audio" in pair:
                waveform = preprocess_audio(audio_data=pair["audio"])
            else:
                continue

            # Extract mel spectrogram features using Whisper's processor
            input_features = processor.feature_extractor(
                waveform, sampling_rate=16000, return_tensors="np"
            ).input_features[0]

            # Tokenize lyrics
            labels = processor.tokenizer(
                pair["lyrics"],
                return_tensors="np",
                padding=False,
            ).input_ids[0]

            sample = {
                "input_features": input_features.tolist(),
                "labels": labels.tolist(),
                "lyrics": pair["lyrics"],
                "song_name": pair.get("song_name", "unknown"),
                "singer": pair.get("singer", "unknown"),
            }
            samples.append(sample)

        # Save split
        save_path = split_dir / "data.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(samples, f)

        print(f"[INFO] {split_name}: {len(samples)} samples saved to {save_path}")

    return output_path


class AutoLyricsDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for loading preprocessed AutoLyrics data."""

    def __init__(self, data_dir: str, split: str = "train"):
        data_path = Path(data_dir) / split / "data.json"
        with open(data_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        print(f"[INFO] Loaded {len(self.samples)} samples from {split} split.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "input_features": torch.tensor(sample["input_features"], dtype=torch.float32),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
        }


def collate_fn(batch, processor):
    """Custom collate function to pad labels to uniform length."""
    input_features = torch.stack([item["input_features"] for item in batch])

    # Pad labels
    label_features = [{"input_ids": item["labels"]} for item in batch]
    labels_batch = processor.tokenizer.pad(
        label_features,
        return_tensors="pt",
        padding=True,
    )

    # Replace padding token id with -100 so it's ignored in loss
    labels = labels_batch["input_ids"].masked_fill(
        labels_batch.attention_mask.ne(1), -100
    )

    return {
        "input_features": input_features,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser(description="AutoLyrics Data Preprocessing")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Root data directory")
    parser.add_argument("--output_dir", type=str, default="./data/processed",
                        help="Output directory for processed data")
    parser.add_argument("--model_name", type=str, default="openai/whisper-base",
                        help="Whisper model name (for processor)")
    parser.add_argument("--no_hf", action="store_true",
                        help="Skip HuggingFace download, use local files only")
    args = parser.parse_args()

    print("=" * 60)
    print("AutoLyrics — Data Preprocessing Pipeline")
    print("=" * 60)

    # Step 1: Download
    print("\n[Step 1] Downloading NUS-48E dataset...")
    raw_dir = download_nus48e(args.data_dir, use_hf=not args.no_hf)

    # Step 2: Find pairs
    print("\n[Step 2] Finding audio-lyric pairs...")
    pairs = find_audio_lyric_pairs(raw_dir)
    if not pairs:
        print("[ERROR] No audio-lyric pairs found. Check your data directory.")
        return

    # Step 3: Preprocess
    print("\n[Step 3] Preprocessing and splitting data...")
    processor = WhisperProcessor.from_pretrained(args.model_name)
    prepare_dataset(pairs, processor, args.output_dir)

    print("\n[DONE] Preprocessing complete!")
    print(f"  Processed data saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
