"""Build PEFT-wrapped Whisper model. Uses 8-bit quantization on CUDA;
falls back to fp32 on CPU (slow — for smoke testing only)."""
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "openai/whisper-small"


def build_processor(model_id: str = MODEL_ID) -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(
        model_id, language="English", task="transcribe"
    )


def build_model_for_training(lora_cfg: dict, model_id: str = MODEL_ID) -> tuple:
    """Return (peft_model, processor) ready for Seq2SeqTrainer.

    On CUDA: loads in 8-bit with BitsAndBytesConfig + Conv1d gradient hook.
    On CPU:  loads in fp32 (8-bit is unsupported without CUDA).
    """
    processor = build_processor(model_id)
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        print("[WARN] CUDA unavailable — loading model in fp32 on CPU. "
              "Training will be very slow; use this only for smoke tests.")
        model = WhisperForConditionalGeneration.from_pretrained(model_id)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # mandatory for gradient checkpointing

    # Conv1d gradient hook: required under 8-bit, harmless otherwise.
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    if use_cuda:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_dropout=lora_cfg.get("lora_dropout", 0.1),
        bias="none",
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "out_proj"],
        ),
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # expect ~0.4–1.0% trainable

    return model, processor
