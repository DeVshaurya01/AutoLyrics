"""Push LoRA adapter to HuggingFace Hub."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export LoRA adapter to HF Hub")
    parser.add_argument("--adapter_path", required=True, help="Local adapter directory")
    parser.add_argument("--repo_id", required=True, help="HF Hub repo, e.g. username/autolyrics-lora")
    parser.add_argument("--private", action="store_true", default=True)
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from huggingface_hub import HfApi

    print(f"[INFO] Pushing adapter from {args.adapter_path} → {args.repo_id}")

    api = HfApi()
    api.upload_folder(
        folder_path=args.adapter_path,
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
    )

    print(f"[DONE] Adapter available at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
