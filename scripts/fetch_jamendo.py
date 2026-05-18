"""Download JamendoLyrics dataset and prepare 30s aligned chunks for training.

JamendoLyrics: 80 CC-licensed songs with word-level synced lyrics.
Repo: https://github.com/f90/jamendolyrics

Pipeline:
  1. git clone the repo (audio + word-level JSON annotations included)
  2. For each song, slice audio into 30s chunks at natural pauses in the
     word-level timing.
  3. Each chunk: audio at 16kHz mono, text = words within its time window.
  4. Save into data/raw/jamendo/ as a parallel raw source.

After this runs, prepare_data_v3.py is updated to also pull from this dir.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
import subprocess
import shutil
from pathlib import Path
import numpy as np
import torch
import torchaudio

JAMENDO_REPO = "https://github.com/f90/jamendolyrics.git"
RAW_DIR = Path("data/raw/jamendo")
SR = 16_000
CHUNK_TARGET_S = 22.0
CHUNK_MAX_S = 28.0
CHUNK_MIN_S = 4.0


def clone_repo():
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        print(f"[INFO] {RAW_DIR} already populated, skipping clone.")
        return
    RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Cloning JamendoLyrics into {RAW_DIR}...")
    subprocess.check_call(
        ["git", "clone", "--depth", "1", JAMENDO_REPO, str(RAW_DIR)]
    )
    print("[INFO] Clone complete.")


def load_audio_16k(path: Path) -> np.ndarray:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=SR)
    return wav.squeeze(0).numpy().astype("float32")


def load_words(json_path: Path) -> list[dict]:
    """Returns [{word, start, end}, ...] from JamendoLyrics word JSON."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    words = []
    # Format varies by jamendolyrics version — handle both
    if isinstance(data, dict) and "words" in data:
        items = data["words"]
    elif isinstance(data, list):
        items = data
    else:
        return []
    for w in items:
        text = w.get("text") or w.get("word") or ""
        start = w.get("start") or w.get("time", 0.0)
        end = w.get("end") or (start + w.get("duration", 0.0))
        if text and end > start:
            words.append({"word": text, "start": float(start), "end": float(end)})
    return words


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def chunk_song(audio: np.ndarray, words: list[dict]) -> list[dict]:
    if not words:
        return []
    chunks = []
    cur_words = []
    cur_start = words[0]["start"]

    for i, w in enumerate(words):
        cur_words.append(w)
        gap_next = words[i + 1]["start"] - w["end"] if i + 1 < len(words) else 999.0
        elapsed = w["end"] - cur_start
        should_break = (elapsed >= CHUNK_TARGET_S and gap_next > 0.4) or elapsed >= CHUNK_MAX_S
        if should_break or i == len(words) - 1:
            chunk_end = w["end"]
            if chunk_end - cur_start >= CHUNK_MIN_S:
                text = " ".join(x["word"] for x in cur_words)
                text = normalize_text(text)
                a0 = int(cur_start * SR)
                a1 = min(int(chunk_end * SR), len(audio))
                audio_slice = audio[a0:a1]
                if text and len(audio_slice) >= int(CHUNK_MIN_S * SR):
                    chunks.append({"audio": audio_slice, "text": text})
            cur_words = []
            cur_start = words[i + 1]["start"] if i + 1 < len(words) else 0.0
    return chunks


def find_word_json(song_dir: Path, song_name: str) -> Path | None:
    """Find the word-level JSON for a song — layout varies by repo version."""
    candidates = [
        song_dir / "annotations" / "words" / f"{song_name}.json",
        song_dir / "lyrics" / f"{song_name}.json",
        song_dir / "words" / f"{song_name}.json",
        song_dir / f"{song_name}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Search anywhere in the repo
    for p in song_dir.rglob(f"{song_name}.json"):
        return p
    return None


def main():
    clone_repo()

    # Find audio files (mp3/wav) and matching word JSONs
    audio_dir_candidates = [
        RAW_DIR / "audios",
        RAW_DIR / "audio",
        RAW_DIR / "mp3",
        RAW_DIR,
    ]
    audio_dir = next((d for d in audio_dir_candidates
                      if d.exists() and any(d.glob("*.mp3")) or any(d.glob("*.wav"))),
                     None)
    if audio_dir is None:
        # Fallback: search anywhere
        audio_files = list(RAW_DIR.rglob("*.mp3")) + list(RAW_DIR.rglob("*.wav"))
        if not audio_files:
            print(f"[ERROR] No audio files found under {RAW_DIR}")
            return
        audio_dir = audio_files[0].parent
        print(f"[INFO] Inferred audio dir: {audio_dir}")

    audio_files = sorted(list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav")))
    print(f"[INFO] Found {len(audio_files)} audio files in {audio_dir}")

    out_dir = RAW_DIR / "_processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    chunk_idx = 0

    for af in audio_files:
        song_name = af.stem
        word_json = find_word_json(RAW_DIR, song_name)
        if word_json is None:
            print(f"  {song_name}: no word JSON, skipping")
            continue
        words = load_words(word_json)
        if not words:
            print(f"  {song_name}: empty word list, skipping")
            continue
        try:
            audio = load_audio_16k(af)
        except Exception as e:
            print(f"  {song_name}: audio load failed ({e})")
            continue
        chunks = chunk_song(audio, words)
        for ch in chunks:
            np.save(out_dir / f"chunk_{chunk_idx:05d}.npy", ch["audio"])
            manifest.append({
                "chunk_idx": chunk_idx,
                "audio_path": str(out_dir / f"chunk_{chunk_idx:05d}.npy"),
                "text": ch["text"],
                "song": song_name,
            })
            chunk_idx += 1
        print(f"  {song_name}: {len(chunks)} chunks")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n[DONE] {len(manifest)} Jamendo chunks saved to {out_dir}")
    print(f"       Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
