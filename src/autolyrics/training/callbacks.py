"""Custom training callbacks."""
import shutil
from pathlib import Path
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments


class CheckpointCleanupCallback(TrainerCallback):
    """Keep only the N best checkpoints by WER to save disk space."""

    def __init__(self, keep_n: int = 3):
        self.keep_n = keep_n

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        output_dir = Path(args.output_dir)
        checkpoints = sorted(
            [d for d in output_dir.iterdir() if d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[-1]),
        )
        for old_ckpt in checkpoints[: -self.keep_n]:
            shutil.rmtree(old_ckpt, ignore_errors=True)
