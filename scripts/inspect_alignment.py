"""Print 10 random training chunks: their reference text + baseline-model hypothesis.

If references are word-misaligned, you'll see things like:
  REF: "every morning you greet me small"
  HYP: "blossom of snow may you bloom"

(model is correctly transcribing the audio, but the ref points at a different
slice of the song — i.e., chunk audio and chunk text are misaligned.)
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import torch
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor

random.seed(0)

ds = load_from_disk("data/processed/hf_dataset")["train"]
N = 10
indices = random.sample(range(len(ds)), N)

print(f"Loading baseline whisper-tiny for sanity-check inference...")
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-tiny", language="English", task="transcribe"
)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

print(f"\n{'='*78}\nInspecting {N} training samples\n{'='*78}\n")

for i, idx in enumerate(indices):
    sample = ds[idx]
    input_features = torch.tensor(sample["input_features"]).to(
        dtype=model.dtype, device=device
    ).unsqueeze(0)
    with torch.no_grad():
        gen = model.generate(
            input_features,
            language="english",
            task="transcribe",
            num_beams=1,
            max_new_tokens=225,
        )
    hyp = processor.tokenizer.decode(gen[0], skip_special_tokens=True).strip()
    label_ids = [t for t in sample["labels"] if t != -100]
    ref = processor.tokenizer.decode(label_ids, skip_special_tokens=True).strip()

    print(f"[{i+1}/{N}] singer={sample.get('singer_id','?')} "
          f"song={sample.get('song_id','?')} mode={sample.get('mode','?')}")
    print(f"  REF: {ref[:200]}")
    print(f"  HYP: {hyp[:200]}")
    print()
