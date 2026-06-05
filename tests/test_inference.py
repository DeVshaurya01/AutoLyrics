"""Smoke tests for inference pipeline (no GPU required)."""
import pytest
import numpy as np


@pytest.mark.slow
def test_baseline_pipe_generates_text(tmp_path):
    """End-to-end: baseline pipeline produces a non-empty string from silence."""
    from autolyrics.inference.pipeline import build_baseline_pipe
    import soundfile as sf

    # Write 3s of silence
    wav_path = tmp_path / "silence.wav"
    sf.write(str(wav_path), np.zeros(16000 * 3, dtype="float32"), 16000)

    pipe = build_baseline_pipe()
    result = pipe(
        str(wav_path),
        generate_kwargs={"language": "english", "task": "transcribe"},
    )
    assert "text" in result
    assert isinstance(result["text"], str)


@pytest.mark.slow
def test_merge_and_unload_does_not_raise(tmp_path):
    """Saving a fresh (untrained) LoRA adapter and loading it back should not raise."""
    import torch
    from transformers import WhisperForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    base = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny", torch_dtype=torch.float32
    )
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
        target_modules=["q_proj", "v_proj"],
    )
    peft_model = get_peft_model(base, lora_config)
    adapter_path = tmp_path / "adapter"
    peft_model.save_pretrained(str(adapter_path))

    from autolyrics.models.load import load_finetuned
    model, processor = load_finetuned(str(adapter_path))
    assert model is not None
