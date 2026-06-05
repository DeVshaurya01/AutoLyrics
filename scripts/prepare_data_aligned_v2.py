"""CTC-aligned dataset prep v2: NUS sung+read for train/val, sung-only for test.

Same singers' spoken audio gives us +100 aligned training chunks with no
distribution shift (same voices) and easier alignment (read speech is
cleaner than sung). Test set stays pure sung.
"""
import os; os.environ["TOKENIZERS_PARALLELISM"]="false"

import re, shutil
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
PROCESSED = Path("data/processed/hf_dataset_v2")
WHISPER_ID = "openai/whisper-small"
W2V_ID = "facebook/wav2vec2-base-960h"
SR = 16_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_SINGERS = ["ADIZ", "JTAN", "MCUR", "MPOL", "MPUR", "NJAT", "PMAR", "ZHIY"]
VAL_SINGERS = ["SAMF", "VKOW"]
TEST_SINGERS = ["JLEE", "KENN"]
CHUNK_TARGET_S = 22.0; CHUNK_MAX_S = 28.0; CHUNK_MIN_S = 4.0


def split_of(s):
    if s in TRAIN_SINGERS: return "train"
    if s in VAL_SINGERS: return "val"
    if s in TEST_SINGERS: return "test"
    return None


def normalize_word(w):
    return re.sub(r"[^A-Z']", "", w.upper())


def align_song(audio, words, m, p):
    norm_words = [normalize_word(w) for w in words]
    keep = [(w, n) for w, n in zip(words, norm_words) if n]
    if not keep: return []
    orig = [k[0] for k in keep]; norms = [k[1] for k in keep]
    vocab = p.tokenizer.get_vocab()
    unk = vocab.get("<unk>", 3)
    targets = []; spans = []
    for n in norms:
        st = len(targets)
        for ch in n:
            targets.append(vocab.get(ch, unk))
        spans.append((st, len(targets)))
    if not targets: return []
    t = torch.tensor([targets], dtype=torch.int32, device=DEVICE)
    inp = p(audio, sampling_rate=SR, return_tensors="pt").input_values.to(DEVICE)
    with torch.no_grad():
        logits = m(inp).logits
    lp = torch.log_softmax(logits, dim=-1)
    il = torch.tensor([lp.size(1)], dtype=torch.int32, device=DEVICE)
    tl = torch.tensor([len(targets)], dtype=torch.int32, device=DEVICE)
    try:
        align, _ = forced_align(lp, t, input_lengths=il, target_lengths=tl, blank=0)
    except Exception:
        return []
    al = align[0].cpu().tolist()
    spf = (audio.shape[0]/lp.size(1)) / SR
    tipf = [-1]*len(al); ct = -1; prev = 0
    for i, tk in enumerate(al):
        if tk == 0: tipf[i] = ct; continue
        if tk != prev: ct += 1
        tipf[i] = ct; prev = tk
    out = []
    for w, (s, e) in zip(orig, spans):
        ff, lf = None, None
        for i, ti in enumerate(tipf):
            if ti >= s and ti < e:
                if ff is None: ff = i
                lf = i
        if ff is None: continue
        out.append((w, ff*spf, (lf+1)*spf))
    return out


def chunk_words(ws, audio):
    if not ws: return []
    chunks = []; cur = [ws[0]]; st = ws[0][1]
    for i in range(1, len(ws)):
        w, s, e = ws[i]
        gap = s - cur[-1][2]; elapsed = e - st
        if (elapsed >= CHUNK_TARGET_S and gap > 0.3) or elapsed >= CHUNK_MAX_S:
            ce = cur[-1][2]
            if ce - st >= CHUNK_MIN_S:
                txt = " ".join(x[0] for x in cur)
                a0 = max(int(st*SR)-int(0.1*SR), 0)
                a1 = min(int(ce*SR)+int(0.1*SR), len(audio))
                if txt: chunks.append({"audio": audio[a0:a1], "text": txt})
            cur = [(w, s, e)]; st = s
        else: cur.append((w, s, e))
    if cur and cur[-1][2]-st >= CHUNK_MIN_S:
        txt = " ".join(x[0] for x in cur)
        a0 = max(int(st*SR)-int(0.1*SR), 0); a1 = min(int(cur[-1][2]*SR)+int(0.1*SR), len(audio))
        if txt: chunks.append({"audio": audio[a0:a1], "text": txt})
    return chunks


def main():
    print("Loading wav2vec2...")
    p = Wav2Vec2Processor.from_pretrained(W2V_ID)
    m = Wav2Vec2ForCTC.from_pretrained(W2V_ID).to(DEVICE).eval()
    lyrics = build_lyrics_map(LYRICS_DIR)
    rows = {"train": [], "val": [], "test": []}

    for sd in sorted(d for d in RAW_DIR.iterdir() if d.is_dir()):
        singer = sd.name
        split = split_of(singer)
        if not split: continue
        # Test: sung only. Train/val: BOTH sung+read
        modes = ["sing"] if split == "test" else ["sing", "read"]
        for mode in modes:
            md = sd / mode
            if not md.is_dir(): continue
            for wav in sorted(md.glob("*.wav")):
                sid = wav.stem
                if sid not in lyrics: continue
                audio = load_wav_16k_mono(wav)
                ws = align_song(audio, lyrics[sid], m, p)
                chs = chunk_words(ws, audio)
                for c in chs:
                    rows[split].append({
                        "audio_array": c["audio"].astype(np.float32),
                        "transcription": c["text"].lower(),
                        "singer_id": singer, "song_id": sid, "mode": mode,
                    })
                print(f"  {singer}/{mode}/{sid}: {len(ws)} words, {len(chs)} chunks")

    print("\nSplit sizes:")
    for k, v in rows.items(): print(f"  {k}: {len(v)}")
    proc = WhisperProcessor.from_pretrained(WHISPER_ID, language="English", task="transcribe")

    def to_hf(s):
        if not s: return None
        d = Dataset.from_dict({
            "audio":[{"array":x["audio_array"],"sampling_rate":SR} for x in s],
            "transcription":[x["transcription"] for x in s],
            "singer_id":[x["singer_id"] for x in s],
            "song_id":[x["song_id"] for x in s],
            "mode":[x["mode"] for x in s],
        })
        return d.cast_column("audio", Audio(sampling_rate=SR))
    splits = {k: to_hf(v) for k, v in rows.items() if v}
    def prep(b):
        a = b["audio"]
        b["input_features"] = proc.feature_extractor(a["array"], sampling_rate=SR).input_features[0]
        b["labels"] = proc.tokenizer(b["transcription"]).input_ids
        return b
    for k in splits:
        splits[k] = splits[k].map(prep, remove_columns=["audio"], num_proc=1)
        before = len(splits[k])
        splits[k] = splits[k].filter(lambda x: len(x["labels"]) <= 448)
        print(f"  {k}: {before} -> {len(splits[k])}")
    if PROCESSED.exists(): shutil.rmtree(PROCESSED)
    DatasetDict(splits).save_to_disk(str(PROCESSED))
    print(f"\n[DONE] {PROCESSED}")


if __name__ == "__main__":
    main()
