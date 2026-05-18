"""AutoLyrics Gradio Demo — Side-by-side Whisper baseline vs LoRA fine-tuned.

Adds:
  - numpy mic input (avoids Gradio filepath/codec freezes on Windows)
  - Optional Demucs vocal isolation toggle (strips instrumental backing)
"""
import os
import traceback
import numpy as np
import torch
import torchaudio
import gradio as gr

from autolyrics.inference.pipeline import build_baseline_pipe, build_finetuned_pipe

ADAPTER_PATH = "outputs/checkpoints/final_adapter"
TARGET_SR = 16000

# Load both pipelines at startup
print("[INFO] Loading baseline pipeline...")
pipe_baseline = build_baseline_pipe()

try:
    print("[INFO] Loading fine-tuned pipeline...")
    pipe_finetuned = build_finetuned_pipe(ADAPTER_PATH)
    _finetuned_available = True
except Exception as e:
    print(f"[WARN] Fine-tuned adapter not found at {ADAPTER_PATH}: {e}")
    print("       Run training first. Baseline-only mode active.")
    pipe_finetuned = None
    _finetuned_available = False

# Lazy-load Demucs on first use (the model file is ~80MB)
_demucs_model = None
def get_demucs():
    global _demucs_model
    if _demucs_model is None:
        try:
            from demucs.pretrained import get_model
            print("[INFO] Loading Demucs vocal-separation model (first time only)...")
            _demucs_model = get_model("htdemucs")
            _demucs_model.eval()
            if torch.cuda.is_available():
                _demucs_model = _demucs_model.cuda()
        except ImportError:
            raise RuntimeError(
                "Demucs not installed. Run: pip install demucs"
            )
    return _demucs_model


def isolate_vocals(audio_np: np.ndarray, sr: int) -> np.ndarray:
    """Run Demucs to extract just the vocal stem from mixed audio.
    Returns mono float32 at TARGET_SR."""
    from demucs.apply import apply_model

    model = get_demucs()
    # Demucs expects (batch, channels, samples) at its native SR (44100)
    if sr != model.samplerate:
        wav = torch.from_numpy(audio_np).float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=model.samplerate)
    else:
        wav = torch.from_numpy(audio_np).float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
    # Demucs needs stereo input — duplicate mono channel
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    wav = wav.unsqueeze(0)
    if torch.cuda.is_available():
        wav = wav.cuda()
    with torch.no_grad():
        sources = apply_model(model, wav, split=True, overlap=0.25, progress=False)
    # sources shape: (batch, n_stems, channels, samples). htdemucs stems order:
    # ["drums", "bass", "other", "vocals"]
    vocals = sources[0, model.sources.index("vocals")].mean(dim=0)
    vocals = vocals.cpu()
    # Resample to TARGET_SR for Whisper
    vocals = torchaudio.functional.resample(
        vocals.unsqueeze(0), orig_freq=model.samplerate, new_freq=TARGET_SR
    ).squeeze(0)
    return vocals.numpy().astype("float32")


def load_audio_any(audio_input) -> tuple[np.ndarray, int]:
    """Accept Gradio audio (sr, np.ndarray) tuple OR filepath. Return (np_float32, sr)."""
    if audio_input is None:
        return None, None
    if isinstance(audio_input, tuple):
        sr, arr = audio_input
        arr = np.asarray(arr)
        # Gradio mic returns int16; convert to float32 in [-1, 1]
        if arr.dtype.kind in "iu":
            max_val = np.iinfo(arr.dtype).max
            arr = arr.astype(np.float32) / max_val
        else:
            arr = arr.astype(np.float32)
        # Stereo -> mono
        if arr.ndim == 2:
            arr = arr.mean(axis=1 if arr.shape[1] < arr.shape[0] else 0)
        return arr, sr
    # Filepath fallback
    wav, sr = torchaudio.load(str(audio_input))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze().numpy().astype("float32"), sr


def transcribe(audio_input, use_vocal_isolation):
    if audio_input is None:
        return "No audio provided.", "No audio provided."

    try:
        arr, sr = load_audio_any(audio_input)
        if arr is None or len(arr) == 0:
            return "Empty audio.", "Empty audio."

        if use_vocal_isolation:
            try:
                arr = isolate_vocals(arr, sr)
                sr = TARGET_SR
            except Exception as e:
                tb = traceback.format_exc()
                print(tb)
                return (f"[Vocal isolation failed] {e}\n\nInstall: pip install demucs",
                        f"[Vocal isolation failed] {e}")

        if sr != TARGET_SR:
            wav_t = torch.from_numpy(arr).float().unsqueeze(0)
            wav_t = torchaudio.functional.resample(wav_t, orig_freq=sr, new_freq=TARGET_SR)
            arr = wav_t.squeeze(0).numpy()

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return f"Audio error: {e}", f"Audio error: {e}"

    # Use greedy (num_beams=1) in the demo for speed. Beam search is used
    # during evaluation for accuracy; the UI prioritizes responsiveness.
    kwargs = {"generate_kwargs": {"language": "english", "task": "transcribe",
                                   "num_beams": 1, "no_repeat_ngram_size": 3},
              "chunk_length_s": 30, "stride_length_s": 5}

    try:
        baseline_text = pipe_baseline(arr, **kwargs)["text"].strip()
    except Exception as e:
        baseline_text = f"[Baseline error] {e}"

    if _finetuned_available:
        try:
            finetuned_text = pipe_finetuned(arr, **kwargs)["text"].strip()
        except Exception as e:
            finetuned_text = f"[Fine-tuned error] {e}"
    else:
        finetuned_text = "Fine-tuned model not available. Train first."

    return baseline_text, finetuned_text


with gr.Blocks(title="AutoLyrics") as demo:
    gr.Markdown(
        "# AutoLyrics\n"
        "### Singing Voice Transcription: Baseline vs LoRA Fine-Tuned\n"
        "Upload a sung audio clip or record via microphone. "
        "Both Whisper-small (zero-shot) and the LoRA fine-tuned variant transcribe it.\n\n"
        "**Tip:** For songs with backing music, enable *Vocal Isolation* below — "
        "it runs Demucs first to strip drums/bass/instruments so only vocals reach the model."
    )

    with gr.Row():
        audio_input = gr.Audio(
            sources=["upload", "microphone"],
            type="numpy",
            label="Upload or Record Audio",
        )

    with gr.Row():
        vocal_iso = gr.Checkbox(
            label="🎤 Vocal Isolation (Demucs)",
            value=False,
            info="Enable for songs with backing music. Slower (~10s extra). "
                 "Disable for a-cappella / clean vocal recordings.",
        )

    btn = gr.Button("Transcribe", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Baseline (Zero-Shot)")
            output_baseline = gr.Textbox(
                label="Whisper-Small (no fine-tuning)", lines=6
            )
        with gr.Column():
            gr.Markdown("### Fine-Tuned (LoRA)")
            output_finetuned = gr.Textbox(
                label="Whisper-Small + LoRA (NUS-48E)", lines=6
            )

    btn.click(
        fn=transcribe,
        inputs=[audio_input, vocal_iso],
        outputs=[output_baseline, output_finetuned],
        api_name=False,
    )

    sample_files = []
    for name in ["sample_sung_01.wav", "sample_sung_02.wav", "sample_spoken_01.wav"]:
        path = os.path.join("examples", name)
        if os.path.exists(path):
            sample_files.append(path)
    if sample_files:
        gr.Examples(examples=sample_files, inputs=audio_input)


if __name__ == "__main__":
    demo.launch(share=False)
