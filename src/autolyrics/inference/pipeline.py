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


def build_baseline_pipe(model_id: str = MODEL_ID):
    """Vanilla Whisper-small in fp16."""
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=DTYPE
    ).to(DEVICE)
    processor = WhisperProcessor.from_pretrained(
        model_id, language="English", task="transcribe"
    )
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=DTYPE,
        device=DEVICE,
    )


def build_finetuned_pipe(adapter_path: str, model_id: str = MODEL_ID):
    """Load base in fp16, attach LoRA adapter, merge and unload.

    Do NOT load in 8-bit for inference — 8-bit Whisper inference is slow
    and produces hallucinations (PEFT Discussion #477).
    """
    base_model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=DTYPE
    )
    # Keep PEFT adapter dynamic instead of merging — avoids any precision
    # mismatch between base weights and LoRA deltas at inference time.
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.to(DEVICE)
    model.config.use_cache = True

    processor = WhisperProcessor.from_pretrained(
        model_id, language="English", task="transcribe"
    )
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=DTYPE,
        device=DEVICE,
    )
