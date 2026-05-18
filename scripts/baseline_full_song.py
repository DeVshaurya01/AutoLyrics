"""Measure the REAL baseline WER on NUS-48E.

Transcribes each full song audio (no chunking) with Whisper-small zero-shot,
compares to canonical lyrics in data/interim/lyrics/. This bypasses the
chunking pipeline entirely so the WER reflects actual model performance
on this dataset, not the alignment quality of our chunker.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import torch
import jiwer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from autolyrics.data.nus48e import load_wav_16k_mono

RAW_DIR = Path("data/raw/nus-smc-corpus_48")
LYRICS_DIR = Path("data/interim/lyrics")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_lyrics_map() -> dict[str, str]:
    """Return {song_id: lyrics_string} — already normalized."""
    out = {}
    for f in sorted(LYRICS_DIR.glob("[0-9][0-9].txt")):
        out[f.stem] = f.read_text(encoding="utf-8").strip()
    return out


def chunk_audio_for_whisper(audio, sr=16000, chunk_s=28, overlap_s=2):
    """Split long audio into <=30s chunks with overlap. Whisper hard-limits at 30s."""
    n = len(audio)
    step = int((chunk_s - overlap_s) * sr)
    size = int(chunk_s * sr)
    chunks = []
    pos = 0
    while pos < n:
        chunks.append(audio[pos:pos + size])
        if pos + size >= n:
            break
        pos += step
    return chunks


def transcribe_full_song(model, processor, audio):
    """Transcribe an arbitrarily long song by chunking + concatenating."""
    parts = []
    for piece in chunk_audio_for_whisper(audio):
        if len(piece) < 16000:  # skip <1s tails
            continue
        feats = processor.feature_extractor(
            piece, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(dtype=model.dtype, device=DEVICE)
        with torch.no_grad():
            ids = model.generate(
                feats,
                language="english",
                task="transcribe",
                num_beams=1,
                max_new_tokens=225,
            )
        parts.append(processor.tokenizer.decode(ids[0], skip_special_tokens=True))
    return " ".join(parts).strip()


def main():
    print(f"Loading Whisper-small on {DEVICE}...")
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="English", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model = model.to(DEVICE).eval()
    normalizer = EnglishTextNormalizer(
        getattr(processor.tokenizer, "english_spelling_normalizer", {}) or {}
    )

    lyrics_map = load_lyrics_map()
    print(f"Loaded canonical lyrics for {len(lyrics_map)} songs.\n")

    rows = []   # (song_id, singer, mode, wer, hyp_first_100, ref_first_100)
    all_refs, all_hyps = [], []

    for singer_dir in sorted(RAW_DIR.iterdir()):
        if not singer_dir.is_dir():
            continue
        singer = singer_dir.name
        for mode in ("sing", "read"):
            mode_dir = singer_dir / mode
            if not mode_dir.is_dir():
                continue
            for wav in sorted(mode_dir.glob("*.wav")):
                song_id = wav.stem
                ref = lyrics_map.get(song_id, "")
                if not ref:
                    continue
                audio = load_wav_16k_mono(wav)
                hyp = transcribe_full_song(model, processor, audio)

                ref_n = normalizer(ref)
                hyp_n = normalizer(hyp)
                if not ref_n.strip():
                    continue
                wer = jiwer.wer(ref_n, hyp_n)
                rows.append((song_id, singer, mode, wer, hyp[:120], ref[:120]))
                all_refs.append(ref_n)
                all_hyps.append(hyp_n)

                print(f"  {singer}/{mode}/{song_id}  WER={wer:.3f}")

    print(f"\n{'='*72}")
    print(f"REAL BASELINE — Whisper-small zero-shot on full songs")
    print(f"{'='*72}")

    overall = jiwer.wer(all_refs, all_hyps)
    print(f"Overall WER (normalized): {overall:.4f}  on {len(all_refs)} songs\n")

    # Aggregate by mode
    sung = [(r, h) for (_, _, m, *_), r, h in zip(rows, all_refs, all_hyps) if m == "sing"]
    read = [(r, h) for (_, _, m, *_), r, h in zip(rows, all_refs, all_hyps) if m == "read"]
    if sung:
        r, h = zip(*sung)
        print(f"  Sung   WER: {jiwer.wer(list(r), list(h)):.4f}  ({len(sung)} samples)")
    if read:
        r, h = zip(*read)
        print(f"  Spoken WER: {jiwer.wer(list(r), list(h)):.4f}  ({len(read)} samples)")

    # Per-song
    per_song = {}
    for (sid, _, _, w, *_), in zip(rows):
        per_song.setdefault(sid, []).append(w)
    print(f"\n  Per-song WER (avg across singers):")
    for sid in sorted(per_song):
        vals = per_song[sid]
        print(f"    {sid}: {sum(vals)/len(vals):.3f}  (n={len(vals)})")

    # Save predictions
    out_dir = Path("outputs/reports/baseline_full_song")
    out_dir.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["song_id", "singer", "mode", "wer", "hyp_preview", "ref_preview"])
        w.writerows(rows)
    print(f"\nFull predictions saved to {out_dir/'predictions.csv'}")


if __name__ == "__main__":
    main()
