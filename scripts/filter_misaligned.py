"""Filter chunks where canonical text doesn't actually match audio content.

Detection: run baseline whisper-small on each chunk, compute WER vs ref.
- WER > 0.9: almost certainly an alignment failure (different content)
- WER 0.5-0.9: ambiguous, could be real hard chunk or partial misalign
- WER < 0.5: alignment is fine

We drop chunks with baseline_wer > 0.9 from ALL splits (uniform filter,
no test-set bias). Saves cleaned dataset back to data/processed/hf_dataset.
"""
import os; os.environ["TOKENIZERS_PARALLELISM"]="false"
import shutil
from pathlib import Path
import torch, jiwer
from datasets import load_from_disk, DatasetDict
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

THRESHOLD = 0.5
MODEL_ID = "openai/whisper-small"
DATASET_IN = Path("data/processed/hf_dataset")
DATASET_OUT = Path("data/processed/hf_dataset_strict")

ds = load_from_disk(str(DATASET_IN))
proc = WhisperProcessor.from_pretrained(MODEL_ID, language="English", task="transcribe")
norm = EnglishTextNormalizer(getattr(proc.tokenizer,"english_spelling_normalizer",{}) or {})
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16).cuda().eval()

def chunk_wer(sample):
    feats = torch.tensor(sample["input_features"]).to(torch.float16).cuda().unsqueeze(0)
    with torch.no_grad():
        ids = model.generate(feats, language="english", task="transcribe",
                             num_beams=1, max_new_tokens=225)
    hyp = norm(proc.tokenizer.decode(ids[0], skip_special_tokens=True))
    ref = norm(proc.tokenizer.decode([t for t in sample["labels"] if t != -100], skip_special_tokens=True))
    if not ref.strip():
        return 0.0
    try:
        return jiwer.wer([ref], [hyp])
    except Exception:
        return 1.0

cleaned = {}
for split in ds:
    print(f"Scoring {split}...")
    wers = []
    for i, s in enumerate(ds[split]):
        wers.append(chunk_wer(s))
        if (i+1) % 30 == 0:
            print(f"  {i+1}/{len(ds[split])}")
    keep_idx = [i for i, w in enumerate(wers) if w <= THRESHOLD]
    dropped = len(wers) - len(keep_idx)
    print(f"  {split}: {len(keep_idx)}/{len(wers)} kept, dropped {dropped}")
    cleaned[split] = ds[split].select(keep_idx)

if DATASET_OUT.exists():
    shutil.rmtree(DATASET_OUT)
DatasetDict(cleaned).save_to_disk(str(DATASET_OUT))
print(f"\n[DONE] Cleaned dataset at {DATASET_OUT}")
for k, v in cleaned.items():
    print(f"  {k}: {len(v)}")
