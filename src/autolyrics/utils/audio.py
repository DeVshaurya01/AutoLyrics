"""Audio I/O helpers."""
import numpy as np
import torchaudio
import torchaudio.transforms as T
from pathlib import Path

TARGET_SR = 16_000


def load_wav_16k_mono(path: str | Path) -> np.ndarray:
    """Load any WAV/MP3, return float32 numpy array at 16 kHz mono."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = T.Resample(orig_freq=sr, new_freq=TARGET_SR)(wav)
    return wav.squeeze(0).numpy().astype("float32")
