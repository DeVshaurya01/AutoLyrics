"""CTC forced-alignment data prep — truly aligned, no drift.

Uses wav2vec2-base-960h + torchaudio.functional.forced_align to align
NUS-48E canonical lyrics to audio at word level. Chunks on word
boundaries — no sp-counting, no whisper bias. Builds train/val/test
all from the SAME alignment process.

Singers (build.md):
  train: ADIZ, JTAN, MCUR, MPOL, MPUR, NJAT, PMAR, ZHIY
  val:   SAMF, VKOW
  test:  JLEE, KENN
Mode: sung only.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import shutil
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torchaudio.functional import forced_align
from datasets import Dataset, DatasetDict, Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, WhisperProcessor

from autolyrics.data.nus48e import load_wav_16k_mono, build_lyrics_map

RAW_DIR = Path("data/raw/nus-smc-corpus_48")
LYRICS_DIR = Path("data/interim/lyrics")
PROCESSED = Path("data/processed/hf_dataset")
WHISPER_ID = "openai/whisper-small"
W2V_ID = "facebook/wav2vec2-base-960h"
SR = 16_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_SINGERS = ["ADIZ", "JTAN", "MCUR", "MPOL", "MPUR", "NJAT", "PMAR", "ZHIY"]
VAL_SINGERS = ["SAMF", "VKOW"]
TEST_SINGERS = ["JLEE", "KENN"]

CHUNK_TARGET_S = 22.0
CHUNK_MAX_S = 28.0
CHUNK_MIN_S = 4.0


def split_of(s):
    if s in TRAIN_SINGERS: return "train"
    if s in VAL_SINGERS: return "val"
    if s in TEST_SINGERS: return "test"
    return None


def normalize_word(w):
    return re.sub(r"[^A-Z']", "", w.upper())


def align_song(audio, words, w2v_model, w2v_proc):
    """Run CTC forced alignment. Returns [(word_original, start_s, end_s), ...]."""
    norm_words = [normalize_word(w) for w in words]
    keep = [(w, n) for w, n in zip(words, norm_words) if n]
    if not keep:
        return []
    orig_words = [k[0] for k in keep]
    norm = [k[1] for k in keep]

    # Build character target sequence + word boundaries
    vocab = w2v_proc.tokenizer.get_vocab()
    # wav2vec2-base-960h vocab: chars A-Z, ', |, <pad>, <unk>, etc.
    unk_id = vocab.get("<unk>", 3)

    targets = []
    word_spans = []  # (start_idx_inclusive, end_idx_exclusive) in targets
    for n in norm:
        start = len(targets)
        for ch in n:
            tid = vocab.get(ch, unk_id)
            targets.append(tid)
        word_spans.append((start, len(targets)))

    if not targets:
        return []
    targets_t = torch.tensor([targets], dtype=torch.int32, device=DEVICE)

    # Run wav2vec2 to get log-probs
    inputs = w2v_proc(audio, sampling_rate=SR, return_tensors="pt").input_values.to(DEVICE)
    with torch.no_grad():
        logits = w2v_model(inputs).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    input_lengths = torch.tensor([log_probs.size(1)], dtype=torch.int32, device=DEVICE)
    target_lengths = torch.tensor([len(targets)], dtype=torch.int32, device=DEVICE)

    try:
        alignments, scores = forced_align(
            log_probs, targets_t,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=0,
        )
    except Exception as e:
        print(f"  forced_align error: {e}")
        return []

    align = alignments[0].cpu().tolist()
    # ratio: audio samples per CTC frame
    samples_per_frame = audio.shape[0] / log_probs.size(1)
    sr_per_frame = samples_per_frame / SR  # seconds per frame

    # For each target index, find first/last frame assigned to it
    # alignments is a tensor of length T (frames) with target indices (or blank)
    # We need to find, for each target position, the range of frames that emit it
    # Standard forced_align returns: for each frame, which target index (or blank=0) emitted
    # Actually torchaudio's forced_align returns alignments of shape (N, T) where each entry
    # is the index INTO TARGETS that this frame is aligned to, OR 0 for blank.

    # Build per-target frame ranges
    target_frames = [[] for _ in range(len(targets))]
    for frame_i, t_idx in enumerate(align):
        # In torchaudio forced_align output, blanks are 0 and non-blank entries are token ids
        # We need to know WHICH target position emitted at this frame
        # Actually torchaudio's forced_align returns the token IDs not target indices.
        # We need merge-by-token approach.
        pass

    # Simpler: re-derive per-target index alignment by walking the path
    # forced_align returns token IDs that match the target sequence after collapsing blanks/repeats
    out_words = []
    target_pos = 0
    cur_first_frame = None
    last_token = None
    for frame_i, tok in enumerate(align):
        if tok == 0:  # blank
            continue
        # advancing through the targets sequence
        if target_pos < len(targets) and tok == targets[target_pos]:
            # New target position reached
            if cur_first_frame is None:
                cur_first_frame = frame_i
            # check if we just stepped to a new target
            if last_token is None or last_token != tok or target_pos == 0:
                # first appearance of this target
                pass
        # Detect transition: when consecutive non-blank frames change to a different token
        # We'll instead walk path more carefully below
        last_token = tok

    # Cleaner: trellis walk to assign each NON-BLANK frame to its sequential target index
    # In CTC forced alignment, the alignment path visits targets in order, possibly with
    # repeats. We can determine target index per frame by counting unique non-blank entries.
    target_idx_per_frame = [-1] * len(align)
    cur_target = -1
    prev_tok = 0
    for i, tok in enumerate(align):
        if tok == 0:
            target_idx_per_frame[i] = cur_target  # stays in current target's span
            continue
        # Non-blank
        if tok != prev_tok:
            # Could be: advance to next target, OR repeat same target's char
            # The CTC rule: same target char gets emitted with blank in between, OR
            # consecutive same char without blank = different target
            cur_target += 1
        target_idx_per_frame[i] = cur_target
        prev_tok = tok

    # Now map word_spans to frame ranges
    for word_orig, (s, e) in zip(orig_words, word_spans):
        # Find first frame where target_idx == s, last frame where target_idx == e-1
        first_f, last_f = None, None
        for i, ti in enumerate(target_idx_per_frame):
            if ti >= s and ti < e:
                if first_f is None:
                    first_f = i
                last_f = i
        if first_f is None or last_f is None:
            continue
        start_s = first_f * sr_per_frame
        end_s = (last_f + 1) * sr_per_frame
        out_words.append((word_orig, start_s, end_s))
    return out_words


def chunk_words(words_ts, audio):
    """Group word-timestamped tuples into ≤max_s chunks at natural gaps."""
    if not words_ts:
        return []
    chunks = []
    cur = [words_ts[0]]
    cur_start = words_ts[0][1]

    for i in range(1, len(words_ts)):
        w, s, e = words_ts[i]
        prev_e = cur[-1][2]
        gap = s - prev_e
        elapsed = e - cur_start
        should_break = (elapsed >= CHUNK_TARGET_S and gap > 0.3) or elapsed >= CHUNK_MAX_S
        if should_break:
            chunk_end = cur[-1][2]
            if chunk_end - cur_start >= CHUNK_MIN_S:
                text = " ".join(x[0] for x in cur).strip()
                a0 = max(int(cur_start * SR) - int(0.1 * SR), 0)
                a1 = min(int(chunk_end * SR) + int(0.1 * SR), len(audio))
                if text:
                    chunks.append({"audio": audio[a0:a1], "text": text,
                                   "start": cur_start, "end": chunk_end})
            cur = [(w, s, e)]
            cur_start = s
        else:
            cur.append((w, s, e))

    # flush
    if cur and (cur[-1][2] - cur_start) >= CHUNK_MIN_S:
        text = " ".join(x[0] for x in cur).strip()
        a0 = max(int(cur_start * SR) - int(0.1 * SR), 0)
        a1 = min(int(cur[-1][2] * SR) + int(0.1 * SR), len(audio))
        if text:
            chunks.append({"audio": audio[a0:a1], "text": text,
                           "start": cur_start, "end": cur[-1][2]})
    return chunks


def main():
    print(f"Loading wav2vec2 ({W2V_ID}) on {DEVICE}...")
    w2v_proc = Wav2Vec2Processor.from_pretrained(W2V_ID)
    w2v_model = Wav2Vec2ForCTC.from_pretrained(W2V_ID).to(DEVICE).eval()

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
            if song_id not in lyrics_map:
                continue
            audio = load_wav_16k_mono(wav)
            words_ts = align_song(audio, lyrics_map[song_id], w2v_model, w2v_proc)
            chunks = chunk_words(words_ts, audio)
            for ch in chunks:
                rows[split].append({
                    "audio_array": ch["audio"].astype(np.float32),
                    "transcription": ch["text"].lower(),
                    "singer_id": singer,
                    "song_id": song_id,
                    "mode": "sing",
                })
            print(f"  {singer}/{song_id}: {len(words_ts)} words aligned, {len(chunks)} chunks")

    print("\nSplit sizes:")
    for k, v in rows.items():
        print(f"  {k}: {len(v)}")

    processor = WhisperProcessor.from_pretrained(WHISPER_ID, language="English", task="transcribe")

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
        b = len(splits[k])
        splits[k] = splits[k].filter(lambda x: len(x["labels"]) <= 448)
        print(f"  {k}: {b} -> {len(splits[k])} after long-label filter")

    if PROCESSED.exists():
        shutil.rmtree(PROCESSED)
    DatasetDict(splits).save_to_disk(str(PROCESSED))
    print(f"\n[DONE] Saved to {PROCESSED}")


if __name__ == "__main__":
    main()
