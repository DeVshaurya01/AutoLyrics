"""Gradio Web UI for AutoLyrics."""
import gradio as gr
import os
import tempfile
import traceback
from autolyrics.inference.pipeline import build_baseline_pipe, build_finetuned_pipe
try:
    from scripts.align import align_audio
except ImportError:
    from align import align_audio

# Lazy loading pipes
pipes = {}

def get_pipe(model_type, adapter_path=None):
    key = model_type
    if key not in pipes:
        if model_type == "Baseline":
            pipes[key] = build_baseline_pipe()
        else:
            pipes[key] = build_finetuned_pipe(adapter_path)
    return pipes[key]

def process_audio(audio_path, model_choice):
    if not audio_path:
        return "Please upload an audio file."

    adapter_path = "outputs/checkpoints/final_adapter"
    if model_choice == "Fine-Tuned AutoLyrics" and not os.path.exists(adapter_path):
        return "Fine-Tuned model adapter not found. Please complete training first."

    try:
        pipe = get_pipe(model_choice, adapter_path)

        # Use a unique temp file path; close handle before align_audio writes to it (Windows)
        tmp = tempfile.NamedTemporaryFile(suffix=".lrc", delete=False)
        lrc_path = tmp.name
        tmp.close()

        align_audio(pipe, audio_path, lrc_path)

        if not os.path.exists(lrc_path) or os.path.getsize(lrc_path) == 0:
            return f"[ERROR] align_audio produced no output at {lrc_path}"

        with open(lrc_path, "r", encoding="utf-8") as f:
            lrc_text = f.read()

        if not lrc_text.strip():
            return "[ERROR] Output file was empty. Model may have transcribed nothing."

        return lrc_text
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)  # also dump to terminal
        return f"[ERROR] {type(e).__name__}: {e}\n\n{tb}"

def build_ui():
    with gr.Blocks(title="CC-AutoLyrics") as app:
        gr.Markdown("# 🎤 AutoLyrics: Whisper Fine-Tuned on NUS-48E")
        gr.Markdown("Upload a sung audio file to automatically generate synced lyrics (LRC format).")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(type="filepath", label="Upload Song")
                model_dropdown = gr.Dropdown(
                    choices=["Baseline", "Fine-Tuned AutoLyrics"],
                    value="Fine-Tuned AutoLyrics",
                    label="Model Selection"
                )
                submit_btn = gr.Button("Generate Lyrics")
            
            with gr.Column():
                lyrics_output = gr.TextArea(label="Generated LRC Lyrics", lines=20)
                
        submit_btn.click(
            fn=process_audio,
            inputs=[audio_input, model_dropdown],
            outputs=lyrics_output
        )
        
    return app

if __name__ == "__main__":
    app = build_ui()
    app.launch()
