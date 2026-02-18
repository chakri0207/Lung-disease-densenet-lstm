from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from pathlib import Path
from typing import Any, List, Tuple

import gradio as gr
from PIL import Image

from src.inference import load_predictor
from src.config import BEST_MODEL_PATH, CONFIG_JSON_PATH


# Load once at startup
PREDICTOR = load_predictor(ckpt_path=BEST_MODEL_PATH, cfg_path=CONFIG_JSON_PATH)


def gradio_predict(audio_path: str) -> Tuple[str, Image.Image | None, List[List[Any]] | None]:
    if audio_path is None or str(audio_path).strip() == "":
        return "Please upload/record an audio file.", None, None

    try:
        pred_label, plot_img, table = PREDICTOR.predict_file(audio_path)
        top3 = ", ".join([f"{lbl} ({p:.2f})" for lbl, p in table[:3]])

        pred_text = (
            f"**Prediction:** {pred_label}\n\n"
            f"**Top-3:** {top3}\n\n"
            "⚠️ Educational demo only — not medical advice."
        )
        return pred_text, plot_img, table

    except Exception as e:
        return f"Error: {e}", None, None


def build_ui():
    title = "Lung Sound Classification (6 classes) — DenseNet121 + BiLSTM"
    desc = (
        "Upload or record a lung sound. The model predicts one of the trained classes.\n\n"
        "⚠️ This is an educational student project demo, **not** a medical device."
    )

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(desc)

        with gr.Row():
            audio = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Lung sound input",
            )

        btn = gr.Button("Predict")
        out_text = gr.Markdown()
        out_plot = gr.Image(label="Probabilities (bar chart)")
        out_table = gr.Dataframe(headers=["Label", "Probability"], datatype=["str", "number"], interactive=False)

        btn.click(fn=gradio_predict, inputs=[audio], outputs=[out_text, out_plot, out_table])

        gr.Markdown(
            "### Notes\n"
            "- Best results come from clean recordings (minimal noise).\n"
            "- Audio is standardized (pad/crop) and split into segments for the LSTM.\n"
            "- Model + config are loaded from `models/`.\n"
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(show_error=True, share=True)
