"""Alignment engine: Transcribe audio with word-level timestamps and output an .lrc file."""
import argparse
import json
import os
from pathlib import Path
import torch

from autolyrics.inference.pipeline import build_finetuned_pipe, build_baseline_pipe

def format_timestamp(seconds: float) -> str:
    """Format seconds into LRC timestamp format [mm:ss.xx]."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    # We use 2 decimals for LRC
    hundredths = int(round((seconds - int(seconds)) * 100))
    # Handle rollover if rounding to 100
    if hundredths == 100:
        hundredths = 0
        secs += 1
        if secs == 60:
            secs = 0
            minutes += 1
    return f"[{minutes:02d}:{secs:02d}.{hundredths:02d}]"

def align_audio(pipe, audio_path: str, output_path: str):
    """Run pipeline with word-level timestamps and generate LRC."""
    print(f"Processing: {audio_path}")
    import torchaudio
    import torchaudio.transforms as T
    
    # Load audio with torchaudio (bypasses ffmpeg requirement in pipeline)
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert stereo to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Resample to 16000Hz (Whisper's expected sample rate)
    if sample_rate != 16000:
        resampler = T.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        
    # The pipeline expects a numpy array
    audio_array = waveform.squeeze().numpy()
    
    # Run the pipeline
    out = pipe(audio_array, return_timestamps="word", chunk_length_s=30)
    chunks = out.get("chunks", [])
    if not chunks:
        print("No timestamp chunks returned by the model.")
        return

    # In Whisper, chunks contain 'text' and 'timestamp' (start, end)
    # Since we want a standard .lrc file, we usually group words into lines
    # For simplicity, we can output one word per line or group by a short pause
    
    lines = []
    current_line_words = []
    current_line_start = None
    
    for chunk in chunks:
        word = chunk["text"].strip()
        ts = chunk["timestamp"] # (start, end)
        
        if ts[0] is None:
            continue
            
        if current_line_start is None:
            current_line_start = ts[0]
            
        current_line_words.append(word)
        
        # Simple heuristic: end the line if the word ends with punctuation or it's a long gap
        # Or we can just output every ~5-7 words
        if len(current_line_words) >= 6 or word.endswith(('.', ',', '?', '!')):
            lines.append((current_line_start, " ".join(current_line_words)))
            current_line_words = []
            current_line_start = None

    # Handle remaining words
    if current_line_words:
        lines.append((current_line_start, " ".join(current_line_words)))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"[ti:Unknown]\\n")
        f.write(f"[ar:Unknown]\\n")
        f.write(f"[by:AutoLyrics]\\n")
        for start_time, text in lines:
            ts_str = format_timestamp(start_time)
            f.write(f"{ts_str} {text}\\n")
            
    print(f"Saved LRC to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Align lyrics using fine-tuned model")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument("--output", required=True, help="Path to output .lrc file")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter. If not provided, uses baseline.")
    args = parser.parse_args()

    if args.adapter and os.path.exists(args.adapter):
        print(f"Loading fine-tuned model with adapter: {args.adapter}")
        pipe = build_finetuned_pipe(args.adapter)
    else:
        print("Loading baseline model...")
        pipe = build_baseline_pipe()

    align_audio(pipe, args.audio, args.output)

if __name__ == "__main__":
    main()
