"""Main training entry point."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

import argparse
from pathlib import Path
import torch
from datasets import load_from_disk
from omegaconf import OmegaConf

from autolyrics.models.whisper_lora import build_model_for_training
from autolyrics.training.trainer import build_trainer
from autolyrics.utils.paths import DATA_DIR, OUTPUTS_DIR, CONFIGS_DIR


def main():
    parser = argparse.ArgumentParser(description="Train Whisper+LoRA on NUS-48E")
    parser.add_argument(
        "--dataset",
        default=str(DATA_DIR / "processed" / "hf_dataset"),
        help="Path to HF dataset produced by prepare_data.py",
    )
    parser.add_argument(
        "--lora_config",
        default="default",
        choices=["default", "decoder_only"],
        help="Which LoRA config to use",
    )
    parser.add_argument(
        "--output_dir",
        default=str(OUTPUTS_DIR / "checkpoints"),
    )
    args = parser.parse_args()

    # Load configs
    lora_cfg = OmegaConf.to_container(
        OmegaConf.load(CONFIGS_DIR / "lora" / f"{args.lora_config}.yaml"),
        resolve=True,
    )
    training_cfg = OmegaConf.to_container(
        OmegaConf.load(CONFIGS_DIR / "training" / "default.yaml"),
        resolve=True,
    )
    training_cfg["output_dir"] = args.output_dir

    # On CPU, disable mixed-precision and gradient checkpointing (CUDA-only paths)
    if not torch.cuda.is_available():
        training_cfg["bf16"] = False
        training_cfg["fp16"] = False
        training_cfg["gradient_checkpointing"] = False
        # Smaller batch + no eager evals so a CPU smoke test finishes
        training_cfg["per_device_train_batch_size"] = 1
        training_cfg["per_device_eval_batch_size"] = 1
        training_cfg["gradient_accumulation_steps"] = 1
        training_cfg["max_steps"] = 1
        training_cfg["eval_strategy"] = "no"
        training_cfg["save_strategy"] = "no"
        training_cfg["load_best_model_at_end"] = False
        training_cfg["report_to"] = []
        print("[WARN] CPU smoke-test mode: 1 step, no eval, no checkpointing.")

    print("=" * 60)
    print("AutoLyrics — Training")
    print(f"LoRA config  : {args.lora_config}")
    print(f"Dataset      : {args.dataset}")
    print(f"Output dir   : {args.output_dir}")
    print("=" * 60)

    ds = load_from_disk(args.dataset)

    model, processor = build_model_for_training(lora_cfg)

    trainer = build_trainer(model, processor, ds, training_cfg)

    print("\n[INFO] Starting training...")
    trainer.train()

    final_adapter = Path(args.output_dir) / "final_adapter"
    model.save_pretrained(str(final_adapter))
    processor.save_pretrained(str(final_adapter))
    print(f"\n[DONE] Adapter saved to {final_adapter}")


if __name__ == "__main__":
    main()
