"""Load baseline or fine-tuned (merged) Whisper models for inference."""
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

MODEL_ID = "openai/whisper-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def load_baseline(model_id: str = MODEL_ID):
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=DTYPE
    ).to(DEVICE)
    processor = WhisperProcessor.from_pretrained(
        model_id, language="English", task="transcribe"
    )
    return model, processor


def load_finetuned(adapter_path: str, model_id: str = MODEL_ID):
    """Load base in fp16, attach LoRA adapter, merge and unload.

    Never load in 8-bit for inference — causes hallucinations (PEFT #477).
    """
    base = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=DTYPE
    )
    # Keep PEFT adapter dynamic instead of merging — avoids precision drift
    # between base weights and LoRA deltas during generation.
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.to(DEVICE)
    model.config.use_cache = True  # re-enable for inference speed

    processor = WhisperProcessor.from_pretrained(
        model_id, language="English", task="transcribe"
    )
    return model, processor
