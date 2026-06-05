"""Test eval: baseline whisper-small vs LoRA adapter."""
import os; os.environ["TOKENIZERS_PARALLELISM"]="false"
import json, sys
from pathlib import Path
import torch, jiwer
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from peft import PeftModel

MODEL_ID = "openai/whisper-small"
ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "outputs/checkpoints/final_adapter"

ds = load_from_disk('data/processed/hf_dataset')['test']
proc = WhisperProcessor.from_pretrained(MODEL_ID, language='English', task='transcribe')
norm = EnglishTextNormalizer(getattr(proc.tokenizer,'english_spelling_normalizer',{}) or {})

def eval_m(model, name):
    refs, hyps = [], []
    for s in ds:
        feats = torch.tensor(s['input_features']).to(model.dtype).cuda().unsqueeze(0)
        with torch.no_grad():
            ids = model.generate(feats, language='english', task='transcribe',
                                 num_beams=5, no_repeat_ngram_size=3, max_new_tokens=225)
        hyps.append(proc.tokenizer.decode(ids[0], skip_special_tokens=True).strip())
        refs.append(proc.tokenizer.decode([t for t in s['labels'] if t != -100], skip_special_tokens=True).strip())
    rn=[norm(r) for r in refs]; hn=[norm(h) for h in hyps]
    pairs=[(r,h) for r,h in zip(rn,hn) if r.strip()]
    wer=jiwer.wer([p[0] for p in pairs],[p[1] for p in pairs])
    cer=jiwer.cer([p[0] for p in pairs],[p[1] for p in pairs])
    print(f'[{name}] WER={wer:.4f}  CER={cer:.4f}  n={len(pairs)}')
    return wer,cer,refs,hyps

print('Baseline...')
base = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16).cuda().eval()
b_wer, b_cer, refs, b_hyps = eval_m(base, 'BASE')
del base; torch.cuda.empty_cache()

print('Fine-tuned...')
ftb = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
ft = PeftModel.from_pretrained(ftb, ADAPTER).cuda().eval()
ft.config.use_cache = True
f_wer, f_cer, _, f_hyps = eval_m(ft, 'FT')

rel = (b_wer - f_wer) / b_wer * 100
print(f'\n=== RESULT (adapter: {ADAPTER}) ===')
print(f'Baseline  WER {b_wer:.4f}  CER {b_cer:.4f}')
print(f'Finetuned WER {f_wer:.4f}  CER {f_cer:.4f}')
print(f'Relative WER reduction: {rel:.2f}%   (target >=15%)')

out = {'adapter': str(ADAPTER), 'baseline_wer': b_wer, 'baseline_cer': b_cer,
       'finetuned_wer': f_wer, 'finetuned_cer': f_cer, 'relative_wer_reduction_pct': rel,
       'n': len(ds)}
Path('outputs/reports').mkdir(parents=True, exist_ok=True)
Path('outputs/reports/final_results.json').write_text(json.dumps(out, indent=2, default=str))
rows = [{'ref':r,'base':b,'ft':f} for r,b,f in zip(refs, b_hyps, f_hyps)]
Path('outputs/reports/final_predictions.json').write_text(json.dumps(rows, indent=2))
