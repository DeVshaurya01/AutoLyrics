"""Raw NUS-48E → HuggingFace DatasetDict preprocessing pipeline."""
import os
import json
import soundfile as sf
from pathlib import Path
from datasets import Dataset, DatasetDict, Audio

from autolyrics.data.nus48e import (
    load_wav_16k_mono,
    parse_nus_label,
    build_lyrics_map,
    chunk_by_silence,
    singer_to_split,
    TARGET_SR,
)
from autolyrics.utils.paths import DATA_DIR


def build_manifests(raw_dir: Path, lyrics_dir: Path) -> dict[str, list[dict]]:
    """Walk NUS-48E corpus and return split→[sample] manifests."""
    lyrics_map = build_lyrics_map(lyrics_dir)
    manifests: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for singer_dir in sorted(raw_dir.iterdir()):
        if not singer_dir.is_dir():
            continue
        singer_id = singer_dir.name
        try:
            split = singer_to_split(singer_id)
        except ValueError:
            continue

        for mode in ("sing", "read"):
            mode_dir = singer_dir / mode
            if not mode_dir.exists():
                continue

            for wav_path in sorted(mode_dir.glob("*.wav")):
                song_id = wav_path.stem  # "01" … "20"
                label_path = wav_path.with_suffix(".txt")

                if not label_path.exists():
                    print(f"[WARN] No label file for {wav_path}, skipping.")
                    continue

                if song_id not in lyrics_map:
                    print(f"[WARN] No canonical lyrics for song {song_id}, skipping.")
                    continue

                audio = load_wav_16k_mono(wav_path)
                segments = parse_nus_label(label_path)
                lyric_words = lyrics_map[song_id]

                chunks = chunk_by_silence(audio, segments, lyric_words)
                for chunk in chunks:
                    manifests[split].append({
                        "audio_array": chunk["audio"].tolist(),
                        "transcription": chunk["text"],
                        "singer_id": singer_id,
                        "song_id": song_id,
                        "mode": mode,
                        "start": chunk["start"],
                        "end": chunk["end"],
                    })

    return manifests


def save_manifests(manifests: dict[str, list[dict]], manifest_dir: Path) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split, samples in manifests.items():
        out = manifest_dir / f"{split}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(samples, f)
        print(f"[INFO] {split}: {len(samples)} chunks → {out}")


def save_audio_chunks(manifests: dict[str, list[dict]], processed_dir: Path) -> dict[str, list[dict]]:
    """Write audio arrays to WAV files, replace array with file path."""
    updated: dict[str, list[dict]] = {}
    for split, samples in manifests.items():
        split_dir = processed_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        updated_samples = []
        for i, sample in enumerate(samples):
            wav_path = split_dir / f"chunk_{i:05d}.wav"
            audio = sample["audio_array"]
            import numpy as np
            sf.write(str(wav_path), np.array(audio, dtype="float32"), TARGET_SR)
            updated_samples.append({
                "audio_path": str(wav_path),
                "transcription": sample["transcription"],
                "singer_id": sample["singer_id"],
                "song_id": sample["song_id"],
                "mode": sample["mode"],
            })
        updated[split] = updated_samples
    return updated


def build_hf_dataset(
    manifests: dict[str, list[dict]],
    processor,
    max_label_len: int = 448,
) -> DatasetDict:
    """Convert manifests to a HuggingFace DatasetDict with mel features + labels."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def prepare_sample(batch):
        import numpy as np
        audio = batch["audio"]["array"]
        sr = batch["audio"]["sampling_rate"]
        feats = processor.feature_extractor(
            audio, sampling_rate=sr, return_tensors="np"
        ).input_features[0]
        ids = processor.tokenizer(batch["transcription"]).input_ids
        batch["input_features"] = feats
        batch["labels"] = ids
        return batch

    splits = {}
    for split, samples in manifests.items():
        ds = Dataset.from_dict({
            "audio": [s["audio_path"] for s in samples],
            "transcription": [s["transcription"] for s in samples],
            "singer_id": [s["singer_id"] for s in samples],
            "song_id": [s["song_id"] for s in samples],
            "mode": [s["mode"] for s in samples],
        })
        ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
        ds = ds.map(prepare_sample, remove_columns=["audio"])
        ds = ds.filter(lambda x: len(x["labels"]) <= max_label_len)
        splits[split] = ds

    return DatasetDict(splits)
