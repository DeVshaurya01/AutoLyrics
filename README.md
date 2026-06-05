# AutoLyrics

Fine-tune **OpenAI Whisper** with **LoRA adapters** so it transcribes *sung* vocals (not just speech), then run inference over full songs to produce time-aligned `.lrc` / `.srt` lyric files. Trained and evaluated on the **NUS-48E** sung-speech corpus, with optional Jamendo augmentation.

A Gradio web demo ships in [app.py](app.py) for drag-and-drop lyric alignment.

---

## Why this project exists

Off-the-shelf Whisper is trained on read/spoken speech. It collapses on singing because of held vowels, melisma, vibrato, wide pitch range, and music-bed interference. AutoLyrics adapts Whisper to the singing domain with parameter-efficient fine-tuning (LoRA on attention + MLP projections) so we don't have to retrain the full 244M / 769M-parameter encoder-decoder — only a few million adapter weights.

**Current baseline (Whisper-tiny, untuned, v3 eval):** WER 0.1872 on test chunks
**Current fine-tuned (LoRA r=32, 25 epochs):** WER 0.1771 → ~5.38% relative reduction
**Target:** >15% relative WER reduction (Currently retraining on `whisper-small` with Jamendo augmentation, LoRA r=16, α=32, for 50 epochs to reach target)

---

## Project layout

```
AutoLyrics/
├── src/autolyrics/          # library code
│   ├── data/                # NUS-48E loading, chunking, HF dataset builder
│   ├── models/              # Whisper + LoRA wrappers, adapter load/save
│   ├── training/            # Seq2Seq trainer, collators, metrics
│   ├── evaluation/          # WER/CER, normalization, per-song reports
│   ├── inference/           # full-song pipeline, VAD, chunk stitching
│   └── utils/               # logging, paths, audio helpers
├── scripts/                 # CLI entrypoints (see "Scripts" below)
├── configs/                 # Hydra config tree
│   ├── base.yaml
│   ├── data/    nus48e.yaml
│   ├── model/   whisper_small.yaml, whisper_medium.yaml
│   ├── lora/    default.yaml, decoder_only.yaml
│   ├── training/ default.yaml
│   └── evaluation/ default.yaml
├── data/
│   ├── raw/      nus-smc-corpus_48/, jamendo/        # you provide
│   ├── interim/  lyrics/, manifests, chunked audio
│   └── processed/hf_dataset/                          # HF Arrow dataset
├── outputs/
│   ├── checkpoints/  LoRA adapters (incl. final_adapter/)
│   ├── reports/      WER/CER, per-song breakdowns
│   ├── logs/         tensorboard
│   └── wandb/        optional W&B runs
├── tests/                   # pytest suite
├── app.py                   # Gradio demo
├── Makefile                 # convenience targets
└── requirements.txt
```

---

## Install

First, clone the repository from GitHub:

```bash
git clone https://github.com/DeVshaurya01/AutoLyrics.git
cd AutoLyrics
```

Requires **Python ≥ 3.10** and (for training) a **CUDA GPU** — `bitsandbytes` 4-bit quantization is used to fit Whisper-small/medium with headroom for activations.

```bash
make install
# equivalent to:
pip install -e . -r requirements.txt
```

Key pinned dependencies (see [requirements.txt](requirements.txt)):

| Purpose            | Package · version              |
|--------------------|--------------------------------|
| Core ML            | torch 2.4.1, transformers 4.46.3, accelerate 1.1.1 |
| PEFT + quant       | peft 0.13.2, bitsandbytes 0.44.1 |
| Audio              | librosa 0.10.2, soundfile 0.12.1 |
| Eval               | jiwer 3.0.5                    |
| Config             | hydra-core 1.3.2, omegaconf 2.3.0 |
| Logging            | tensorboard 2.18.0, wandb 0.18.7 |
| Demo               | gradio 4.44.1                  |

CPU-only setups can still run inference and the Gradio demo (slowly); training is GPU-only in practice.

---

## Data

### NUS-48E (required)

1. Obtain the NUS Sung and Spoken Lyrics Corpus (NUS-48E) and place it at:
   ```
   data/raw/nus-smc-corpus_48/
   ```
2. Provide lyrics transcripts in [data/interim/lyrics/](data/interim/lyrics/) as `01.txt` … `20.txt`, one song per file. A song-list manifest lives in that folder's README.

The split policy is **speaker-disjoint** — no singer appears in more than one split:

| Split | Singers                                       |
|-------|-----------------------------------------------|
| train | ADIZ, JTAN, MCUR, MPOL, MPUR, NJAT, PMAR, ZHIY |
| val   | SAMF, VKOW                                    |
| test  | JLEE, KENN                                    |

Audio is resampled to **16 kHz mono** and chunked to a **22 s target / 28 s max / 3 s min** window with lyric-aligned boundaries (see [configs/data/nus48e.yaml](configs/data/nus48e.yaml)).

### Jamendo (optional augmentation)

Use [scripts/fetch_jamendo.py](scripts/fetch_jamendo.py) to pull a Creative Commons subset into `data/raw/jamendo/` for domain mixing.

---

## Usage

The v3 standard pipeline (updated for `whisper-small` + Jamendo augmentation):

```bash
# 1. Fetch Jamendo augmentation data (saves 500+ chunks to data/raw/jamendo/_processed)
python scripts/fetch_jamendo.py

# 2. Prepare the v3 dataset (NUS-48E + Jamendo)
python scripts/prepare_data_v3.py

# 3. Train the LoRA adapter (default: 50 epochs, r=16, alpha=32)
python scripts/train.py

# 4. Evaluate on the matched-distribution v3 test set
python scripts/evaluate_v3.py

# 5. Run the Gradio UI demo
python app.py
```

### Configs (Hydra)

All knobs live in [configs/](configs/). Override on the CLI by group:

```bash
# Larger model + decoder-only LoRA
python scripts/train.py model=whisper_medium lora=decoder_only

# Bigger batch, shorter run
python scripts/train.py training.per_device_train_batch_size=16 training.num_train_epochs=10
```

Defaults you should know about:

- **Model:** `openai/whisper-small`, English, task=transcribe
- **LoRA:** r=16, α=32, dropout=0.1, targets `q/k/v/out_proj` + `fc1/fc2` in **both encoder and decoder** — fc1/fc2 give the model real capacity to adapt to singing acoustics, not just attention re-routing
- **Training:** 50 epochs, lr 7e-5, cosine schedule, 100 warmup steps, bf16, gradient checkpointing, eval every 50 steps, best-by-WER checkpoint kept
- **Generation:** `predict_with_generate=true`, max length 225

### Single-song alignment

```bash
python scripts/align.py --audio path/to/song.wav --adapter outputs/checkpoints/final_adapter --out song.lrc
```

Produces `.lrc` (line-level timestamps) and/or `.srt` from the inference pipeline in [src/autolyrics/inference/pipeline.py](src/autolyrics/inference/pipeline.py).

---

## Scripts

Beyond the four `make` targets, [scripts/](scripts/) contains:

| Script                       | What it does                                                       |
|------------------------------|---------------------------------------------------------------------|
| `prepare_data.py`            | Canonical preprocessing → HF dataset                                |
| `prepare_data_v2.py` / `v3.py` | Alternative chunking / labeling strategies (experimental)        |
| `train.py`                   | Hydra-driven Seq2Seq trainer with LoRA                              |
| `evaluate.py`                | WER/CER on test split, baseline vs. fine-tuned                      |
| `evaluate_v2.py` / `v3.py`   | Variants matching the v2/v3 prep pipelines                          |
| `baseline_full_song.py`      | Runs un-fine-tuned Whisper end-to-end as a reference                |
| `align.py`                   | Full-song inference → `.lrc` / `.srt`                               |
| `app.py` *(scripts/)*        | Same Gradio app, runnable from the scripts dir                      |
| `identify_songs.py`          | Maps NUS-48E filenames → titles                                     |
| `auto_fill_lyrics.py` / `_fill_lyrics.py` | Lyric-import helpers for `data/interim/lyrics/`        |
| `inspect_alignment.py`       | Debug a single alignment (token timings vs. ground truth)           |
| `fetch_jamendo.py`           | Download Jamendo CC subset                                          |
| `export_adapter.py`          | Save a deployable LoRA adapter from a checkpoint                    |
| `final_report.py`            | Aggregate every reports/ run into a single summary                  |

---

## Outputs

- **`outputs/checkpoints/`** — `checkpoint-*/` step checkpoints + `final_adapter/` (load with `peft.PeftModel.from_pretrained`)
- **`outputs/reports/baseline/`** and **`outputs/reports/finetuned/`** — `metrics.json`, `per_song.csv`, predicted vs. reference text dumps
- **`outputs/logs/`** — TensorBoard scalars (`tensorboard --logdir outputs/logs`)
- **`*.lrc` / `*.srt`** — produced by `scripts/align.py` or the Gradio demo

---

## Demo

**[Watch the AutoLyrics Demo Video](https://drive.google.com/file/d/1UuEqjQFU5i8JaYTTbLc3j-4MX12PD-gM/view)**

```bash
make demo   # or: python app.py
```

Opens a Gradio UI: drop in an audio file, get back a synced `.lrc` preview plus downloadable `.lrc` and `.srt`. The demo loads `outputs/checkpoints/final_adapter` if present, otherwise falls back to base Whisper.

---

## Status & known issues

- Initial fine-tune lands ~4% relative WER reduction — below the >15% target. Likely levers: more epochs on Whisper-medium, Jamendo augmentation, lyric-cleanup pass on `data/interim/lyrics/`.
- HF Transformers warns about `forced_decoder_ids` colliding with `task=transcribe` at eval time; cosmetic, ignored in favor of `task=transcribe`.
- Flash-attention is not built into the pinned PyTorch wheel on Windows — falls back to `scaled_dot_product_attention`. Functional, just slower.

---

## Testing

```bash
make test
```

The suite (in [tests/](tests/)) covers data loading, chunking math, the inference pipeline, and adapter round-tripping. Tests do **not** require the NUS-48E corpus — they use synthetic audio fixtures.

---

## License & credits

- Built on [OpenAI Whisper](https://github.com/openai/whisper) via Hugging Face Transformers
- LoRA via [PEFT](https://github.com/huggingface/peft)
- Dataset: NUS Sung and Spoken Lyrics Corpus (NUS-48E) — Duan et al., 2013
- Optional augmentation: Jamendo Creative Commons catalogue
