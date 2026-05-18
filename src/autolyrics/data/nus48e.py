"""NUS-48E corpus loader and chunker."""
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path

TARGET_SR = 16_000
CHUNK_TARGET_S = 22.0
CHUNK_MAX_S = 28.0
CHUNK_MIN_S = 3.0

SINGERS = [
    "ADIZ", "JLEE", "JTAN", "KENN", "MCUR", "MPOL",
    "MPUR", "NJAT", "PMAR", "SAMF", "VKOW", "ZHIY",
]

TRAIN_SINGERS = ["ADIZ", "JTAN", "MCUR", "MPOL", "MPUR", "NJAT", "PMAR", "ZHIY"]
VAL_SINGERS   = ["SAMF", "VKOW"]
TEST_SINGERS  = ["JLEE", "KENN"]


def load_wav_16k_mono(path: str | Path) -> np.ndarray:
    """Load any WAV/MP3 and return float32 numpy array at 16 kHz mono."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = T.Resample(orig_freq=sr, new_freq=TARGET_SR)(wav)
    return wav.squeeze(0).numpy().astype("float32")


def parse_nus_label(label_path: str | Path) -> list[tuple[float, float, str]]:
    """Parse NUS-48E label file → [(start_s, end_s, phone), ...].

    NUS-48E uses whitespace-separated columns (not tab as some docs claim).
    Format: "<start_seconds> <end_seconds> <phone_label>"
    """
    segments = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    segments.append((float(parts[0]), float(parts[1]), parts[2]))
                except ValueError:
                    continue
    return segments


def build_lyrics_map(lyrics_dir: str | Path) -> dict[str, list[str]]:
    """Load canonical lyrics for all 20 songs. Returns {song_id: [word, ...]}."""
    lyrics_dir = Path(lyrics_dir)
    lyrics_map = {}
    for f in sorted(lyrics_dir.glob("*.txt")):
        song_id = f.stem  # "01", "02", ..., "20"
        text = f.read_text(encoding="utf-8").strip()
        lyrics_map[song_id] = text.split()
    return lyrics_map


def chunk_by_silence(
    audio: np.ndarray,
    segments: list[tuple[float, float, str]],
    lyric_words: list[str],
    sr: int = TARGET_SR,
    target_s: float = CHUNK_TARGET_S,
    max_s: float = CHUNK_MAX_S,
    min_s: float = CHUNK_MIN_S,
) -> list[dict]:
    """Split audio into ≤max_s chunks at sil/sp boundaries.

    Returns list of {"audio": np.ndarray, "text": str, "start": float, "end": float}.
    Word assignment: sp markers in the phone tier advance the word index.
    """
    chunks = []
    chunk_start = None
    word_idx = 0
    chunk_words: list[str] = []

    for start, end, phone in segments:
        ph = phone.strip().lower()

        if ph in ("sil", "sp"):
            # sp marks a word boundary — advance word pointer
            if ph == "sp" and word_idx < len(lyric_words):
                if chunk_start is not None:
                    chunk_words.append(lyric_words[word_idx])
                word_idx += 1

            # Break if we've hit the target duration
            if chunk_start is not None and (start - chunk_start) >= target_s:
                chunk_audio = audio[int(chunk_start * sr):int(start * sr)]
                if len(chunk_audio) / sr >= min_s:
                    chunks.append({
                        "audio": chunk_audio,
                        "text": " ".join(chunk_words),
                        "start": chunk_start,
                        "end": start,
                    })
                chunk_start = None
                chunk_words = []
            continue

        # Non-silence phone — start a new chunk if needed
        if chunk_start is None:
            chunk_start = start

        # Force-break at max length
        if end - chunk_start >= max_s:
            chunk_audio = audio[int(chunk_start * sr):int(end * sr)]
            if len(chunk_audio) / sr >= min_s:
                chunks.append({
                    "audio": chunk_audio,
                    "text": " ".join(chunk_words),
                    "start": chunk_start,
                    "end": end,
                })
            chunk_start = None
            chunk_words = []

    # Flush remaining audio
    if chunk_start is not None and chunk_words:
        remaining = audio[int(chunk_start * sr):]
        if len(remaining) / sr >= min_s:
            chunks.append({
                "audio": remaining,
                "text": " ".join(chunk_words),
                "start": chunk_start,
                "end": len(audio) / sr,
            })

    return chunks


def singer_to_split(singer_id: str) -> str:
    """Map a singer code to its train/val/test split."""
    if singer_id in TRAIN_SINGERS:
        return "train"
    if singer_id in VAL_SINGERS:
        return "val"
    if singer_id in TEST_SINGERS:
        return "test"
    raise ValueError(f"Unknown singer: {singer_id}")
