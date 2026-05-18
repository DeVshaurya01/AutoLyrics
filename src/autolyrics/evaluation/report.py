"""Per-singer, per-song, and sung-vs-spoken WER breakdown reports."""
import json
import csv
from pathlib import Path
from collections import defaultdict

import jiwer
from transformers import WhisperTokenizer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

_normalizer = None


def _get_normalizer():
    global _normalizer
    if _normalizer is None:
        tok = WhisperTokenizer.from_pretrained("openai/whisper-small")
        mapping = getattr(tok, "english_spelling_normalizer", {}) or {}
        _normalizer = EnglishTextNormalizer(mapping)
    return _normalizer


def _wer_pair(refs: list[str], hyps: list[str]) -> dict:
    n = _get_normalizer()
    norm_refs = [n(r) for r in refs]
    norm_hyps = [n(h) for h in hyps]
    pairs = [(r, h) for r, h in zip(norm_refs, norm_hyps) if r.strip()]
    if not pairs:
        return {"wer": None, "cer": None, "n": 0}
    r_list, h_list = zip(*pairs)
    return {
        "wer": round(jiwer.wer(list(r_list), list(h_list)), 4),
        "cer": round(jiwer.cer(list(r_list), list(h_list)), 4),
        "n": len(pairs),
    }


def generate_report(
    predictions: list[dict],  # [{ref, hyp, singer_id, song_id, mode}, ...]
    output_dir: str | Path,
    baseline_wer: float | None = None,
) -> dict:
    """Compute four breakdown axes and write JSON, Markdown, and CSV reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    refs = [p["ref"] for p in predictions]
    hyps = [p["hyp"] for p in predictions]

    results = {}

    # 1. Overall
    results["overall"] = _wer_pair(refs, hyps)

    # 2. Per singer
    by_singer: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for p in predictions:
        by_singer[p["singer_id"]][0].append(p["ref"])
        by_singer[p["singer_id"]][1].append(p["hyp"])
    results["per_singer"] = {
        sid: _wer_pair(r, h) for sid, (r, h) in by_singer.items()
    }

    # 3. Per song
    by_song: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for p in predictions:
        by_song[p["song_id"]][0].append(p["ref"])
        by_song[p["song_id"]][1].append(p["hyp"])
    results["per_song"] = {
        sid: _wer_pair(r, h) for sid, (r, h) in by_song.items()
    }

    # 4. Sung vs spoken
    by_mode: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for p in predictions:
        by_mode[p["mode"]][0].append(p["ref"])
        by_mode[p["mode"]][1].append(p["hyp"])
    results["sung_vs_spoken"] = {
        mode: _wer_pair(r, h) for mode, (r, h) in by_mode.items()
    }

    # Relative WER reduction (vs baseline)
    if baseline_wer is not None and results["overall"]["wer"] is not None:
        rel = (baseline_wer - results["overall"]["wer"]) / baseline_wer * 100
        results["relative_wer_reduction_pct"] = round(rel, 2)

    # Write JSON
    json_path = output_dir / "eval_results.json"
    json_path.write_text(json.dumps(results, indent=2))

    # Write Markdown
    md_lines = ["# AutoLyrics Evaluation Report\n"]
    overall = results["overall"]
    md_lines.append(f"## Overall\n- WER (normalized): {overall['wer']}\n- CER: {overall['cer']}\n- Samples: {overall['n']}\n")
    if "relative_wer_reduction_pct" in results:
        md_lines.append(f"- Relative WER reduction vs baseline: **{results['relative_wer_reduction_pct']}%**\n")
    md_lines.append("\n## Sung vs Spoken\n")
    for mode, m in results["sung_vs_spoken"].items():
        md_lines.append(f"- **{mode}**: WER={m['wer']}, CER={m['cer']}, n={m['n']}\n")
    md_lines.append("\n## Per Singer\n")
    for sid, m in sorted(results["per_singer"].items()):
        md_lines.append(f"- **{sid}**: WER={m['wer']}, CER={m['cer']}, n={m['n']}\n")
    md_lines.append("\n## Per Song\n")
    for song, m in sorted(results["per_song"].items()):
        md_lines.append(f"- **Song {song}**: WER={m['wer']}, CER={m['cer']}, n={m['n']}\n")
    (output_dir / "report.md").write_text("".join(md_lines))

    # Write predictions CSV for qualitative inspection
    csv_path = output_dir / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["singer_id", "song_id", "mode", "ref", "hyp"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"[INFO] Reports written to {output_dir}")
    return results
