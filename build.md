# AutoLyrics — Implementation-Ready Technical Blueprint
### Whisper + LoRA Fine-Tuning for Singing Voice Transcription on NUS-48E

> **Purpose**: This document is a strict, file-by-file specification for an AI coding agent to build the entire AutoLyrics project from scratch. Every config value, module name, and code pattern is pinned and verified against primary sources.

---

## 0. Executive Decision Summary

| Decision | Value | Rationale |
|---|---|---|
| Base model | `openai/whisper-small` (244M params) | Whisper overfits on <10h data with larger models; `small` is the HuggingFace-canonical choice per Sanchit Gandhi's blog |
| LoRA targets | **Both encoder + decoder** attention: `q_proj`, `k_proj`, `v_proj`, `out_proj` | Singing is an *acoustic* OOD shift — the encoder must adapt to sustained vowels and pitch contours. Gao et al. (2024/2025) and LoRA-Whisper papers target encoder modules for music. Decoder-only variant kept as ablation |
| LoRA rank | `r=8`, `lora_alpha=16`, `lora_dropout=0.1` | Conservative rank to prevent overfitting on 169 min of audio; scale to `r=16` only if training loss plateaus |
| Quantization | 8-bit via BitsAndBytes (training only) | 4-bit NF4 is less stable for Whisper; fp16 at inference (never 8-bit — causes hallucinations per PEFT Discussion #477) |
| Dataset split | Speaker-disjoint: 8 train / 2 val / 2 test singers | Prevents lyric leakage; gender-balanced |
| Chunking | 20–25s segments at `sil`/`sp` boundaries | Whisper hard-caps at 30s; silence-aligned cuts preserve word boundaries |
| Jiwer version | Pin `jiwer==3.0.5` | `jiwer>=4.0` removed `compute_measures`, breaking `evaluate.load("wer")` — confirmed in HuggingFace evaluate Issue #684 |
| Gradio version | `gradio==4.44.1` (matches your linked docs) | Stable API; `gr.Audio(sources=["upload","microphone"], type="filepath")` |
| Evaluation target | >15% relative WER reduction on sung audio | `(WER_baseline − WER_finetuned) / WER_baseline × 100` |

---

## 1. Complete Directory Structure

```
autolyrics/
├── app.py                              # Gradio demo entry point (HF Spaces convention)
├── packages.txt                        # System deps for Spaces: ffmpeg, libsndfile1
├── requirements.txt                    # Pinned Python deps (see §8)
├── pyproject.toml                      # src-layout package metadata
├── README.md
├── LICENSE
├── Makefile                            # Convenience targets: prepare, train, eval, demo
├── .gitignore
│
├── configs/                            # Hydra-style YAML configs
│   ├── base.yaml                       # defaults list, paths, seed=42
│   ├── data/
│   │   └── nus48e.yaml                 # corpus path, sample_rate=16000, split policy
│   ├── model/
│   │   ├── whisper_small.yaml          # model_id, mel_bins=80, max_length=448
│   │   └── whisper_medium.yaml         # fallback candidate
│   ├── lora/
│   │   ├── default.yaml                # r=8, alpha=16, both enc+dec
│   │   └── decoder_only.yaml           # ablation: decoder attention only
│   ├── training/
│   │   └── default.yaml                # lr=1e-4, epochs=10, batch=8, grad_accum=2
│   └── evaluation/
│       └── default.yaml                # metrics, normalization, generation params
│
├── data/
│   ├── raw/                            # .gitignored — drop NUS-48E corpus here
│   │   └── nus-smc-corpus_48/          # {SingerCode}/{sing|read}/{SongID}.{wav,txt}
│   ├── interim/
│   │   ├── lyrics/                     # 20 canonical song lyrics: 01.txt … 20.txt
│   │   └── manifests/                  # train/val/test JSON manifests
│   └── processed/                      # Chunked WAVs + HuggingFace Arrow shards
│       ├── train/
│       ├── val/
│       └── test/
│
├── src/autolyrics/
│   ├── __init__.py
│   ├── config.py                       # OmegaConf loader + path resolution
│   ├── data/
│   │   ├── __init__.py
│   │   ├── nus48e.py                   # Parse labels, load audio, chunk by silence
│   │   ├── preprocess.py               # Raw → HuggingFace DatasetDict pipeline
│   │   └── collator.py                 # DataCollatorSpeechSeq2SeqWithPadding + SpecAugment
│   ├── models/
│   │   ├── __init__.py
│   │   ├── whisper_lora.py             # Build PEFT-wrapped Whisper with BnB 8-bit
│   │   └── load.py                     # Load baseline / load fine-tuned + merge_and_unload
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                  # Seq2SeqTrainer wiring with PEFT compatibility flags
│   │   └── callbacks.py                # Early stopping, WER logging, checkpoint cleanup
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                  # compute_metrics() + Whisper EnglishTextNormalizer
│   │   └── report.py                   # Per-singer, per-song, sung-vs-spoken breakdown
│   ├── inference/
│   │   ├── __init__.py
│   │   └── pipeline.py                 # Build HF ASR pipeline with merged LoRA weights
│   └── utils/
│       ├── __init__.py
│       ├── audio.py                    # load_wav_16k_mono, specaugment helpers
│       ├── logging.py                  # WandB / TensorBoard setup
│       └── paths.py                    # Project root resolution
│
├── scripts/
│   ├── prepare_data.py                 # Entry: raw NUS-48E → data/processed/ + Arrow shards
│   ├── train.py                        # Entry: Hydra config → Seq2SeqTrainer.train()
│   ├── evaluate.py                     # Entry: baseline + finetuned → reports
│   └── export_adapter.py              # Push LoRA adapter to HF Hub
│
├── notebooks/
│   ├── 01_eda.ipynb                    # NUS-48E exploration: durations, singer stats
│   ├── 02_baseline_wer.ipynb           # Zero-shot Whisper on sung vs spoken
│   └── 03_results_analysis.ipynb       # Post-training WER breakdowns + visualizations
│
├── outputs/                            # .gitignored
│   ├── checkpoints/                    # LoRA adapter saves (~5-20MB each)
│   ├── logs/                           # TensorBoard event files
│   ├── wandb/                          # WandB run directories
│   └── reports/                        # eval_results.json, report.md, predictions.csv
│
├── examples/                           # 3-5 sample NUS-48E clips for Gradio demo
│   ├── sample_sung_01.wav
│   ├── sample_sung_02.wav
│   └── sample_spoken_01.wav
│
└── tests/
    ├── test_data_parsing.py            # Label file parsing, chunk determinism
    ├── test_collator.py                # Pad masking, BOS stripping
    ├── test_metrics.py                 # WER/CER computation correctness
    └── test_inference.py               # End-to-end 1-step train + predict smoke test
```

---

## 2. Data Pipeline Architecture

### 2.1 NUS-48E Dataset Structure (Verified from Duan et al., APSIPA 2013)

The corpus ships as `{SingerCode}/{sing|read}/{SongID}.{wav,txt}`:

- **12 singers**: `ADIZ, JLEE, JTAN, KENN, MCUR, MPOL, MPUR, NJAT, PMAR, SAMF, VKOW, ZHIY` (6 female / 6 male)
- **20 songs** numbered `01`–`20`, each singer sang 4 songs
- **48 sung + 48 spoken recordings** ≈ 115 min sung + 54 min spoken
- **Audio format**: 44.1 kHz, 16-bit, mono WAV
- **Label files** (`.txt`): Audacity label-track format — **NOT Praat TextGrid**

Each `.txt` line: `<start_seconds>\t<end_seconds>\t<phone_label>`
Phone set: CMU 39-phoneme set + `sil` (silence) and `sp` (short pause / word boundary)

**Critical insight**: There are NO word-level or line-level transcriptions shipped. The `.txt` files contain only phone-level timing. Word-level text must be reconstructed from the 20 canonical song lyrics.

### 2.2 Lyrics Reconstruction Strategy

Since Whisper requires orthographic word transcripts, you must:

1. **Manually transcribe the 20 canonical lyrics** into `data/interim/lyrics/{01..20}.txt` — one word per line or space-delimited
2. **Map phone sequences to word boundaries** using `sp` markers in the label files — each `sp` roughly corresponds to a word boundary
3. **Alternative**: Run Montreal Forced Aligner on the audio + lyrics to get word-level TextGrids automatically

### 2.3 Speaker-Disjoint Split Policy

```yaml
# configs/data/nus48e.yaml
split_policy: "speaker_disjoint"
train_singers: [ADIZ, JTAN, MCUR, MPOL, MPUR, NJAT, PMAR, ZHIY]  # 8 singers
val_singers:   [SAMF, VKOW]                                        # 2 singers
test_singers:  [JLEE, KENN]                                         # 2 singers
# Gender-balanced: each split has equal M/F representation
```

### 2.4 Audio Preprocessing + Chunking Pipeline

```
scripts/prepare_data.py execution flow:

1. Iterate raw/{singer}/{sing|read}/{song_id}.wav
2. Load WAV → torchaudio.load() → resample 44100→16000 Hz
3. Parse corresponding .txt label file → list of (start, end, phone)
4. Load canonical lyrics from data/interim/lyrics/{song_id}.txt
5. chunk_by_silence():
   - Walk phone segments, accumulate audio
   - Break at sil/sp boundaries once ≥ 20s accumulated
   - Cap chunks at 28s (leave 2s headroom below Whisper's 30s limit)
   - Minimum chunk length: 3s (discard shorter fragments)
   - Track word indices via sp-marker counting to assign lyric slices
6. Save each chunk as: data/processed/{split}/chunk_{idx}.wav
7. Build manifest JSON with columns:
   {audio_path, transcription, singer_id, song_id, mode: "sing"|"read"}
8. Convert to HuggingFace DatasetDict:
   Dataset.from_dict({...}).cast_column("audio", Audio(sampling_rate=16000))
9. Apply prepare_dataset() map function:
   - feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
   - tokenizer(transcription).input_ids → stored as "labels"
   - Filter: discard samples where len(labels) > 448
10. ds.save_to_disk("data/processed/hf_dataset")
```

**Environment variable required**: `os.environ["TOKENIZERS_PARALLELISM"] = "false"` before any `map(num_proc=...)` call to prevent tokenizer fork deadlocks.

### 2.5 Core Data Module: `src/autolyrics/data/nus48e.py`

```python
"""NUS-48E corpus loader and chunker."""
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path

TARGET_SR = 16_000
CHUNK_TARGET_S = 22.0
CHUNK_MAX_S = 28.0
CHUNK_MIN_S = 3.0


def load_wav_16k_mono(path: str | Path) -> np.ndarray:
    """Load any WAV/MP3 and return float32 numpy array at 16kHz mono."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = T.Resample(orig_freq=sr, new_freq=TARGET_SR)(wav)
    return wav.squeeze(0).numpy().astype("float32")


def parse_nus_label(label_path: str | Path) -> list[tuple[float, float, str]]:
    """Parse Audacity-format label file → [(start_s, end_s, phone), ...]."""
    segments = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                segments.append((float(parts[0]), float(parts[1]), parts[2]))
    return segments


def build_lyrics_map(lyrics_dir: str | Path) -> dict[str, list[str]]:
    """Load canonical lyrics for all 20 songs. Returns {song_id: [word, ...]}."""
    lyrics_dir = Path(lyrics_dir)
    lyrics_map = {}
    for f in sorted(lyrics_dir.glob("*.txt")):
        song_id = f.stem  # "01", "02", ..., "20"
        text = f.read_text().strip()
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
    chunk_words = []

    for start, end, phone in segments:
        ph = phone.strip().lower()

        if ph in ("sil", "sp"):
            # sp marks a word boundary — advance word pointer
            if ph == "sp" and word_idx < len(lyric_words):
                if chunk_start is not None:
                    chunk_words.append(lyric_words[word_idx])
                word_idx += 1

            # Check if we should break here
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

    # Flush remaining
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
```

### 2.6 Data Collator: `src/autolyrics/data/collator.py`

Based directly on the canonical pattern from the HuggingFace fine-tune-whisper blog, with optional SpecAugment for training:

```python
"""Speech-Seq2Seq data collator with optional SpecAugment."""
from dataclasses import dataclass
from typing import Any
import torch
import torchaudio.transforms as T


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    apply_specaug: bool = False  # Enable for training batches only

    def __call__(self, features: list[dict]) -> dict:
        # Pad input features (mel spectrograms)
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Optional SpecAugment on training batches
        if self.apply_specaug:
            batch["input_features"] = self._specaugment(batch["input_features"])

        # Pad labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Mask padding tokens with -100 for loss computation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Strip BOS token if prepended by tokenizer
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

    @staticmethod
    def _specaugment(mel: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment: 2× time masks + 2× freq masks."""
        # mel shape: (batch, n_mels=80, time=3000)
        for _ in range(2):
            mel = T.TimeMasking(time_mask_param=40)(mel)
        for _ in range(2):
            mel = T.FrequencyMasking(freq_mask_param=10)(mel)
        return mel
```

**Augmentation policy notes**:
- SpecAugment is the right default — Whisper was pretrained with it
- Mild additive noise (SNR 10–20 dB) and ±10% time-stretch are safe extras
- **NEVER use pitch-shift augmentation**: pitch is content-bearing in singing
- NUS-48E is a-cappella, so no source separation needed

---

## 3. Baseline Evaluation Plan (Zero-Shot Inference)

### 3.1 Step-by-Step Logic for `scripts/evaluate.py`

```
1. Load test split from data/processed/hf_dataset
2. Load baseline model: WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
3. Load processor: WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")
4. For each test sample:
   a. Extract input_features via feature_extractor
   b. Generate with: model.generate(
        input_features,
        language="english",
        task="transcribe",
        num_beams=1,             # greedy for speed; beam=5 for final numbers
        max_new_tokens=225
      )
   c. Decode predicted token IDs → hypothesis string
   d. Retrieve ground-truth transcription → reference string
5. Normalize BOTH reference and hypothesis through Whisper's EnglishTextNormalizer:
   - Lowercases
   - Expands contractions (I'm → i am)
   - Spells out numbers
   - Strips punctuation
6. Compute metrics via jiwer (pinned to 3.0.5):
   - wer_normalized = jiwer.wer(normalized_refs, normalized_hyps)
   - cer_normalized = jiwer.cer(normalized_refs, normalized_hyps)
   - wer_ortho      = jiwer.wer(raw_refs, raw_hyps)
   - cer_ortho      = jiwer.cer(raw_refs, raw_hyps)
7. Repeat steps 2-6 with fine-tuned LoRA model (merged to fp16)
8. Compute relative WER reduction:
   relative_reduction = (wer_baseline - wer_finetuned) / wer_baseline * 100
9. Write outputs/reports/eval_results.json + report.md
```

### 3.2 Evaluation Metrics Module: `src/autolyrics/evaluation/metrics.py`

```python
"""WER/CER computation with Whisper-standard normalization."""
import jiwer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()


def compute_metrics(pred) -> dict:
    """Plugs into Seq2SeqTrainer as compute_metrics callback.

    NOTE: Uses jiwer 3.0.5 API. If upgrading to 4.x, replace
    jiwer.wer(truth=...) with jiwer.wer(reference=...).
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 (pad mask) with pad_token_id for decoding
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Normalized metrics (paper-comparable)
    pred_norm = [normalizer(p) for p in pred_str]
    label_norm = [normalizer(l) for l in label_str]

    # Filter empty references after normalization
    pairs = [(r, h) for r, h in zip(label_norm, pred_norm) if r.strip()]
    if not pairs:
        return {"wer": 1.0, "cer": 1.0, "wer_ortho": 1.0, "cer_ortho": 1.0}

    refs, hyps = zip(*pairs)
    wer_norm = jiwer.wer(list(refs), list(hyps))
    cer_norm = jiwer.cer(list(refs), list(hyps))

    # Raw orthographic metrics
    wer_ortho = jiwer.wer(label_str, pred_str)
    cer_ortho = jiwer.cer(label_str, pred_str)

    return {
        "wer": round(wer_norm, 4),
        "cer": round(cer_norm, 4),
        "wer_ortho": round(wer_ortho, 4),
        "cer_ortho": round(cer_ortho, 4),
    }
```

### 3.3 Detailed Report Axes (`src/autolyrics/evaluation/report.py`)

The evaluation report must include four breakdowns (unique to NUS-48E's design):

1. **Overall**: WER/CER across entire test set (sung + spoken)
2. **Per-singer**: WER for each of the 2 test singers → detect singer-specific difficulty
3. **Per-song**: WER for each song → detect song-specific difficulty (tempo, lyrics complexity)
4. **Sung vs. Spoken**: WER on `mode=sing` vs `mode=read` → quantify the "singing gap" that fine-tuning should close

Output format: `outputs/reports/eval_results.json` + `outputs/reports/report.md` + `outputs/reports/predictions.csv` (ref/hyp pairs for qualitative inspection).

---

## 4. Fine-Tuning Strategy

### 4.1 Model Loading with PEFT + BitsAndBytes: `src/autolyrics/models/whisper_lora.py`

```python
"""Build PEFT-wrapped Whisper model with 8-bit quantization for training."""
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "openai/whisper-small"


def build_processor():
    return WhisperProcessor.from_pretrained(
        MODEL_ID, language="English", task="transcribe"
    )


def build_model_for_training(lora_cfg: dict) -> tuple:
    """Returns (peft_model, processor).

    Critical implementation details (from Vaibhavs10 notebook + PEFT examples):
    1. load_in_8bit via BitsAndBytesConfig — NOT the deprecated load_in_8bit kwarg
    2. Conv1d forward hook on encoder stem — required under 8-bit to propagate gradients
    3. forced_decoder_ids=None — prevents stale language/task forcing
    4. use_cache=False — incompatible with gradient checkpointing
    """
    processor = build_processor()

    # 8-bit quantization config
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Override generation config
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # MANDATORY for gradient checkpointing

    # ══════════════════════════════════════════════════════════════════
    # CRITICAL: Register forward hook on encoder Conv1d stem.
    # Under 8-bit loading, the Conv1d layers lose gradient flow.
    # Without this hook, training silently fails (loss doesn't decrease).
    # Source: Vaibhavs10/fast-whisper-finetuning notebook
    # ══════════════════════════════════════════════════════════════════
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    # Prepare for k-bit training (freeze layers, cast norms to float32)
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_dropout=lora_cfg.get("lora_dropout", 0.1),
        bias="none",
        task_type="SEQ_2_SEQ_LM",
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "out_proj"]
        ),
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expected output: ~0.4-1.0% trainable parameters

    return model, processor
```

### 4.2 LoRA Config Files

**Primary config** (`configs/lora/default.yaml`) — encoder + decoder:
```yaml
r: 8
lora_alpha: 16
lora_dropout: 0.1
bias: "none"
task_type: "SEQ_2_SEQ_LM"
target_modules: ["q_proj", "k_proj", "v_proj", "out_proj"]
# This targets ALL attention projections in both encoder and decoder
# because PEFT matches by module name suffix across the whole model
```

**Ablation config** (`configs/lora/decoder_only.yaml`):
```yaml
r: 8
lora_alpha: 16
lora_dropout: 0.1
bias: "none"
task_type: "SEQ_2_SEQ_LM"
target_modules: ["q_proj", "k_proj", "v_proj", "out_proj"]
modules_to_save: null
# To restrict to decoder only, use explicit layer paths:
# target_modules: ["model.decoder.layers.*.self_attn.q_proj", ...]
# OR use a custom PEFT filter — see PEFT docs on `target_modules` regex
```

### 4.3 Training Configuration: `configs/training/default.yaml`

```yaml
# Seq2SeqTrainingArguments
output_dir: "outputs/checkpoints"
per_device_train_batch_size: 8
per_device_eval_batch_size: 4
gradient_accumulation_steps: 2        # Effective batch = 16
learning_rate: 1.0e-4                 # Standard for LoRA on Whisper
lr_scheduler_type: "linear"
warmup_steps: 50
num_train_epochs: 10
max_steps: -1

# Precision
bf16: true                            # Use fp16 if on T4/V100 (no bf16 support)
fp16: false

# Evaluation & saving
eval_strategy: "steps"
eval_steps: 50
save_strategy: "steps"
save_steps: 50
save_total_limit: 3
load_best_model_at_end: true
metric_for_best_model: "wer"
greater_is_better: false

# Generation during eval
predict_with_generate: true
generation_max_length: 225

# Gradient checkpointing
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false

# CRITICAL PEFT COMPATIBILITY FLAGS
remove_unused_columns: false          # PeftModel.forward has different signature
label_names: ["labels"]               # Trainer won't find labels without this

# Logging
logging_steps: 10
report_to: ["tensorboard", "wandb"]
run_name: "autolyrics-whisper-small-lora"

# Misc
seed: 42
dataloader_num_workers: 2
push_to_hub: false
```

### 4.4 Training Script: `scripts/train.py`

```python
"""Main training entry point."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from datasets import load_from_disk

from autolyrics.models.whisper_lora import build_model_for_training, build_processor
from autolyrics.data.collator import DataCollatorSpeechSeq2SeqWithPadding
from autolyrics.evaluation.metrics import compute_metrics  # inject processor at module level


def main(cfg):
    # 1. Load preprocessed dataset
    ds = load_from_disk(cfg.data.processed_path)

    # 2. Build model
    model, processor = build_model_for_training(cfg.lora)

    # 3. Build collator (with SpecAugment for training)
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        apply_specaug=True,
    )

    # 4. Training arguments
    training_args = Seq2SeqTrainingArguments(**cfg.training)

    # 5. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,  # For padding
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # 6. Train
    trainer.train()

    # 7. Save adapter (only ~5-20 MB)
    model.save_pretrained(cfg.training.output_dir + "/final_adapter")
    processor.save_pretrained(cfg.training.output_dir + "/final_adapter")
```

### 4.5 Key Training Gotchas Checklist

| Gotcha | Fix | Source |
|---|---|---|
| Conv1d gradient death under 8-bit | `conv1.register_forward_hook(lambda m,i,o: o.requires_grad_(True))` | Vaibhavs10 notebook |
| Trainer strips wrong columns with PeftModel | `remove_unused_columns=False` | PEFT docs |
| Trainer can't find labels column | `label_names=["labels"]` | HuggingFace Trainer + PEFT integration |
| `use_cache=True` crashes with gradient checkpointing | Set `model.config.use_cache = False` during training | Transformers docs |
| `forced_decoder_ids` causes stale language forcing | Set to `None`; pass language/task via `generate_kwargs` | Sanchit Gandhi blog |
| `evaluate.load("wer")` crashes with jiwer ≥ 4.0 | Pin `jiwer==3.0.5` OR call `jiwer.wer()` directly | GH evaluate#684 |
| 8-bit inference is slow + hallucinates | Load fp16 at inference, `merge_and_unload()` LoRA | PEFT Discussion #477 |

---

## 5. UI Deployment Plan (Gradio Demo)

### 5.1 Architecture: `app.py`

The demo provides **side-by-side baseline vs. fine-tuned comparison** using `gr.Blocks` (not `gr.Interface`):

```python
"""AutoLyrics Gradio Demo — Side-by-side Whisper baseline vs LoRA fine-tuned."""
import gradio as gr
from autolyrics.inference.pipeline import build_baseline_pipe, build_finetuned_pipe

# Load both pipelines at startup
pipe_baseline = build_baseline_pipe()
pipe_finetuned = build_finetuned_pipe("outputs/checkpoints/final_adapter")


def transcribe(audio_path: str) -> tuple[str, str]:
    """Run both baseline and fine-tuned models on the same audio."""
    if audio_path is None:
        return "No audio provided.", "No audio provided."

    result_base = pipe_baseline(
        audio_path,
        generate_kwargs={"language": "english", "task": "transcribe"},
        chunk_length_s=30,
        stride_length_s=5,
    )
    result_ft = pipe_finetuned(
        audio_path,
        generate_kwargs={"language": "english", "task": "transcribe"},
        chunk_length_s=30,
        stride_length_s=5,
    )
    return result_base["text"].strip(), result_ft["text"].strip()


# Build UI
with gr.Blocks(title="AutoLyrics") as demo:
    gr.Markdown("# 🎵 AutoLyrics\n### Singing Voice Transcription: Baseline vs LoRA Fine-Tuned")

    with gr.Row():
        audio_input = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",        # Pipeline handles resampling internally
            label="Upload or Record Audio",
        )

    btn = gr.Button("Transcribe", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Baseline (Zero-Shot)")
            output_baseline = gr.Textbox(label="Whisper-Small (no fine-tuning)", lines=6)
        with gr.Column():
            gr.Markdown("### Fine-Tuned (LoRA)")
            output_finetuned = gr.Textbox(label="Whisper-Small + LoRA", lines=6)

    btn.click(
        fn=transcribe,
        inputs=[audio_input],
        outputs=[output_baseline, output_finetuned],
    )

    gr.Examples(
        examples=["examples/sample_sung_01.wav", "examples/sample_sung_02.wav"],
        inputs=audio_input,
    )

if __name__ == "__main__":
    demo.launch()
```

### 5.2 Inference Pipeline: `src/autolyrics/inference/pipeline.py`

```python
"""Build HuggingFace ASR pipelines for baseline and fine-tuned models."""
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)
from peft import PeftModel

MODEL_ID = "openai/whisper-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def build_baseline_pipe():
    """Vanilla Whisper-small in fp16."""
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE
    ).to(DEVICE)
    processor = WhisperProcessor.from_pretrained(
        MODEL_ID, language="English", task="transcribe"
    )
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=DTYPE,
        device=DEVICE,
    )


def build_finetuned_pipe(adapter_path: str):
    """Load base model in fp16, attach LoRA adapter, merge and unload.

    IMPORTANT: Do NOT load in 8-bit for inference.
    8-bit Whisper inference is ~5x slower and produces hallucinations
    (documented in PEFT Discussion #477).
    """
    # Load base in fp16
    base_model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE
    )

    # Attach LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Merge LoRA weights into base model and discard adapter overhead
    model = model.merge_and_unload()
    model = model.to(DEVICE)

    # Re-enable cache for inference speed
    model.config.use_cache = True

    processor = WhisperProcessor.from_pretrained(
        MODEL_ID, language="English", task="transcribe"
    )
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=DTYPE,
        device=DEVICE,
    )
```

### 5.3 HuggingFace Spaces Deployment Files

**`packages.txt`** (system dependencies):
```
ffmpeg
libsndfile1
```

**`requirements.txt`** for Spaces (note: NO bitsandbytes — training only):
```
torch==2.4.1
torchaudio==2.4.1
transformers==4.46.3
accelerate==1.1.1
peft==0.13.2
gradio==4.44.1
jiwer==3.0.5
```

---

## 6. Expected Performance Benchmarks

| Model | NUS-48E Sung WER (est.) | NUS-48E Spoken WER (est.) | Notes |
|---|---|---|---|
| Whisper-small zero-shot | 40–70% | 10–25% | No published benchmark exists on NUS-48E |
| Whisper-small + LoRA (decoder only) | 35–55% | 8–20% | Partial adaptation |
| Whisper-small + LoRA (enc+dec) | **25–50%** | **8–18%** | Target: >15% relative reduction |

The "singing gap" (sung WER minus spoken WER) is the core metric. Fine-tuning should compress this gap significantly while maintaining spoken performance.

---

## 7. File-by-File Build Order for the Coding Agent

Build in this exact sequence to surface issues early. Each step has a **test gate** that must pass before proceeding:

| Order | File | Test Gate |
|---|---|---|
| 1 | `src/autolyrics/utils/paths.py` | Resolves project root correctly |
| 2 | `src/autolyrics/data/nus48e.py` | `parse_nus_label()` on one real .txt file returns correct phones |
| 3 | `data/interim/lyrics/01.txt` … `20.txt` | 20 canonical lyrics files exist, match expected word counts |
| 4 | `scripts/prepare_data.py` | Produces ≥400 chunks in `data/processed/`, no chunk >30s |
| 5 | `src/autolyrics/data/collator.py` | Batch of 4 samples pads correctly, labels have -100 masks |
| 6 | `src/autolyrics/models/whisper_lora.py` | `print_trainable_parameters()` shows ~0.4-1.0% trainable |
| 7 | `src/autolyrics/evaluation/metrics.py` | `jiwer.wer(["hello world"], ["hello duck"])` returns 0.5 |
| 8 | `scripts/train.py` | 1-step training completes without error, loss is finite |
| 9 | `scripts/evaluate.py` | Produces `eval_results.json` with all 4 metric keys |
| 10 | `src/autolyrics/inference/pipeline.py` | `merge_and_unload()` succeeds, generates text from audio |
| 11 | `app.py` | Gradio demo launches, both columns show text output |
| 12 | `tests/` | All pytest tests pass |

---

## 8. Pinned Dependencies (`requirements.txt`)

```
# Core ML
torch==2.4.1
torchaudio==2.4.1
transformers==4.46.3
accelerate==1.1.1
datasets==3.1.0

# PEFT + Quantization
peft==0.13.2
bitsandbytes==0.44.1

# Audio processing
librosa==0.10.2.post1
soundfile==0.12.1

# Evaluation
jiwer==3.0.5

# Configuration
hydra-core==1.3.2
omegaconf==2.3.0

# Logging
tensorboard==2.18.0
wandb==0.18.7

# Demo
gradio==4.44.1

# Data science
numpy==1.26.4
pandas==2.2.3
tqdm==4.67.0
```

**Compatibility notes**:
- `peft>=0.10` required for clean Whisper LoRA merging
- `transformers>=4.40` required for `generate_kwargs={"language":...}` in ASR pipeline
- `jiwer==3.0.5` pinned because `jiwer>=4.0` broke `evaluate.load("wer")`; use `jiwer.wer()` / `jiwer.cer()` directly
- `bitsandbytes` is training-only — omit from Spaces `requirements.txt`
- Python 3.10 or 3.11 + CUDA 12.1

---

## 9. Risk Register

| # | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| R1 | No word-level transcriptions in NUS-48E | Confirmed | High | Build `lyrics_map` from 20 canonical song lyrics; align via `sp`-marker counting or Montreal Forced Aligner |
| R2 | Overfitting on 169 min of audio | High | High | `r=8`, dropout=0.1, SpecAugment, early stopping (patience=3), speaker-disjoint splits |
| R3 | Conv1d gradient death under 8-bit | Confirmed | Critical | `register_forward_hook` on `model.model.encoder.conv1` |
| R4 | `jiwer>=4.0` breaks `evaluate.load("wer")` | Confirmed | Medium | Pin `jiwer==3.0.5` |
| R5 | 8-bit inference hallucinations | Documented | High | fp16 + `merge_and_unload()` at serving time — never 8-bit inference |
| R6 | Trainer column stripping with PeftModel | Documented | Critical | `remove_unused_columns=False`, `label_names=["labels"]` |
| R7 | 15% relative reduction not achieved | Medium | Medium | Escalation path: try `r=16`, add spoken data to training, try `whisper-medium` |

---

## 10. Appendix: NUS-48E Download Instructions

The dataset is not on HuggingFace Hub. Access options:

1. **Google Drive** (community mirror from Amphion toolkit): Folder ID `12pP9uUl0HTVANU3IPLnumTJiRjPtVUMx`
2. **OpenDataLab**: `opendatalab.com/OpenDataLab/NUS-48E`
3. **Direct request**: Email Prof. Ye Wang (`wangye@comp.nus.edu.sg`) at NUS Sound and Music Computing Lab

**Citation required** (academic/non-commercial use):
> Z. Duan, H. Fang, B. Li, K. C. Sim, and Y. Wang, "The NUS Sung and Spoken Lyrics Corpus: A Quantitative Comparison of Singing and Speech," in *Proc. APSIPA ASC*, 2013.

Place the extracted corpus at `data/raw/nus-smc-corpus_48/` and ensure the structure follows `{SingerCode}/{sing|read}/{SongID}.{wav,txt}`.
