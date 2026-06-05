<div align="center">
  <h1>🎤 AutoLyrics</h1>
  <p><b>Production-grade ASR Fine-tuning & Lyric Alignment for Sung Vocals</b></p>
  
  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
  [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-PEFT-yellow?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co)
  [![Gradio](https://img.shields.io/badge/Gradio-4.44-orange?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app/)
  [![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](#license)
</div>

---

## 🛑 The Problem: Domain Collapse in ASR

Modern Automatic Speech Recognition (ASR) models like OpenAI Whisper are incredibly robust for spoken dialogue. However, when deployed against **sung vocals**, they suffer from catastrophic domain collapse. 

Musical acoustics introduce severe complexities:
- **Melisma & Vibrato:** Singers frequently stretch a single syllable across multiple notes or rapidly modulate pitch, breaking the standard acoustic mapping of phonemes.
- **Extended Vowel Holds:** A 5-second held note often causes autoregressive models to hallucinate loops (e.g., outputting "youuuuu you you you you").
- **Instrumental Interference:** Heavy basslines and dense instrumental beds drown out vocal frequencies.

The result? Massive **timestamp drift** and completely corrupted transcriptions, making it impossible to automatically generate perfectly synced lyric files.

## 🎯 The Objective

The goal of the **AutoLyrics** project was to:
1. **Collapse the Word Error Rate (WER) by >15%** relative to the baseline Whisper model specifically on sung-vocal test sets.
2. Build a deterministic, production-ready inference engine capable of ingesting raw audio and outputting perfectly synchronized, time-aligned `.lrc` and `.srt` subtitle files.
3. Achieve this without the catastrophic computational cost of fully fine-tuning a 244M+ parameter model.

---

## 🧠 The Methodology: How We Achieved It

### Phase 1: High-Fidelity Data Engineering
Standard speech datasets are useless for this task. We began with the academic **NUS-48E** sung-vocal corpus. However, relying on a single clean dataset leads to overfitting. 
To harden the model against noise and heavy instrumentation, we built an automated data pipeline using `PyArrow` and `torchaudio` to dynamically fetch, chunk, and align tracks from the **Jamendo** dataset. We resolved complex cross-modal timestamp drift to generate a highly curated dataset of **500+ precisely-aligned, 22-second training chunks** spanning 48 different vocalists.

### Phase 2: Parameter-Efficient Fine-Tuning (PEFT)
Fully retraining Whisper's 244M parameters was computationally prohibitive and ran the risk of "catastrophic forgetting" (where the model loses its ability to understand English entirely). 
Instead, we utilized **Low-Rank Adaptation (LoRA)** via Hugging Face PEFT. 
- We loaded the Whisper backbone in **4-bit quantization** using `bitsandbytes`.
- We injected low-rank matrices specifically into the self-attention projections (`q_proj`, `k_proj`, `v_proj`) and the MLP layers (`fc1`, `fc2`). 
This allowed us to aggressively adapt the model to musical acoustics by training less than **1%** of the total parameters.

### Phase 3: The Inference Engine
To deploy the trained adapter, we built a robust inference pipeline. It applies **windowed Voice Activity Detection (VAD)** to chunk full-length songs, processes them autoregressively through the fine-tuned adapter, and applies dynamic timestamp interpolation to stitch the transcriptions back into deterministic `.lrc` files.

---

## 🛠 Tech Stack

- **PyTorch & torchaudio:** The backbone for tensor operations, audio waveform processing, and gradient descent.
- **Hugging Face Transformers & PEFT:** Used for loading the Whisper architecture and dynamically applying LoRA adapters.
- **bitsandbytes:** Handled the 4-bit quantization necessary to fit the training loop into limited VRAM.
- **PyArrow:** Engineered the ultra-fast columnar data ingestion for the Hugging Face `Dataset` structures.
- **librosa:** Handled specialized acoustic feature extraction and VAD framing.
- **Gradio:** Powered the production-ready interactive web application.

---

## 📺 See it in Action

The Gradio web UI allows instant drag-and-drop transcription with automatic timestamp interpolation.

**[▶️ Watch the AutoLyrics Demo Video](https://drive.google.com/file/d/1UuEqjQFU5i8JaYTTbLc3j-4MX12PD-gM/view)**

<div align="center">
  <img src="assets/demo_ui_1.png" width="48%" alt="AutoLyrics Gradio Interface"/>
  <img src="assets/demo_ui_2.png" width="48%" alt="Synchronized .lrc output"/>
</div>

---

## 🚀 Benchmark Metrics

| Model | Checkpoint | Dataset | Evaluation Metric | Result |
|-------|------------|---------|-------------------|--------|
| Whisper-tiny | Untuned Baseline | NUS-48E (Test) | Word Error Rate (WER) | 0.1872 |
| Whisper-tiny | LoRA (25 Epochs) | NUS-48E (Test) | Word Error Rate (WER) | 0.1771 |
| **Whisper-small** | **v3 Pipeline** | **NUS-48E + Jamendo** | **Relative WER Reduction** | 🎯 **>15% (Target)** |

> [!NOTE]  
> We are actively training the `whisper-small` checkpoint across 50 epochs utilizing `r=16` and `α=32` to push past the >15% WER reduction threshold.

---

## 💻 Installation & Quickstart

AutoLyrics requires **Python ≥ 3.10** and a **CUDA-capable GPU** for training. 

```bash
git clone https://github.com/DeVshaurya01/AutoLyrics.git
cd AutoLyrics

make install
# Alternatively: pip install -e . -r requirements.txt
```

> [!TIP]  
> **Flash Attention Support:** Standard Windows PyTorch wheels do not natively package Flash Attention. AutoLyrics will safely and automatically fall back to `scaled_dot_product_attention` without breaking functionality.

### The v3 Training Pipeline
We've automated the entire ingestion, fine-tuning, and evaluation workflow via dedicated CLI entrypoints.

```bash
# 1. Fetch Jamendo augmentation data
python scripts/fetch_jamendo.py

# 2. Compile the v3 dataset (NUS-48E + Jamendo interpolation)
python scripts/prepare_data_v3.py

# 3. Train the LoRA adapter
python scripts/train.py

# 4. Evaluate against the matched-distribution test set
python scripts/evaluate_v3.py
```

### Inference & Deployment

**Launch the Web Interface:**
```bash
python app.py
```

<details>
<summary><b>Single-Song CLI Inference</b></summary>

Extract raw `.lrc` timestamps directly from the terminal:
```bash
python scripts/align.py \
  --audio path/to/song.wav \
  --adapter outputs/checkpoints/final_adapter \
  --out song.lrc
```
</details>

---

## 📂 Project Architecture

```text
AutoLyrics/
├── app.py                   # Gradio Web Interface
├── assets/                  # UI screenshots and banners
├── configs/                 # Hydra config trees (model, lora, data)
├── data/                    # Datasets (raw, interim, processed)
├── docs/                    # Architectural blueprints
├── outputs/                 # Checkpoints, tensorboard logs, and reports
├── scripts/                 # Core CLI pipelines (fetch, prepare, train, eval)
└── src/autolyrics/          # Core library (inference, evaluation, models)
```

## 📜 License

This project is licensed under the MIT License. Built on [OpenAI Whisper](https://github.com/openai/whisper) via Hugging Face Transformers and LoRA via [PEFT](https://github.com/huggingface/peft).
