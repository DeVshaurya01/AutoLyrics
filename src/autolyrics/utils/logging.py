"""WandB / TensorBoard setup helpers."""
import os


def setup_wandb(run_name: str, project: str = "autolyrics") -> None:
    try:
        import wandb
        wandb.init(project=project, name=run_name)
    except ImportError:
        print("[WARN] wandb not installed; skipping WandB init.")


def disable_tokenizers_parallelism() -> None:
    """Must be called before any HuggingFace tokenizer multiprocessing."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
