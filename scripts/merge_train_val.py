"""Merge cleaned train+val into a single training pool.

Use 10% of the combined pool as 'val' just for early-stopping logging.
Test set stays untouched (speaker-disjoint).
"""
import shutil
from pathlib import Path
from datasets import load_from_disk, concatenate_datasets, DatasetDict

SRC = Path("data/processed/hf_dataset_strict")
OUT = Path("data/processed/hf_dataset_strict_merged")

ds = load_from_disk(str(SRC))
combined = concatenate_datasets([ds["train"], ds["val"]]).shuffle(seed=42)
n = len(combined)
val_n = max(int(0.1 * n), 10)
val = combined.select(range(val_n))
train = combined.select(range(val_n, n))

out = DatasetDict({"train": train, "val": val, "test": ds["test"]})
if OUT.exists():
    shutil.rmtree(OUT)
out.save_to_disk(str(OUT))
print(f"train: {len(train)}, val: {len(val)}, test: {len(ds['test'])}")
print(f"Saved to {OUT}")
