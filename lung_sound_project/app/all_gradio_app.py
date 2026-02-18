"""
Gradio demo app for Lung Sound Classification (6 classes)
Model A (Mel-only): DenseNet121 + BiLSTM (PyTorch)

Expected project layout:
lung_sound_project/
  models/
    best_model.pth        # (recommended) focal-trained checkpoint renamed as best_model.pth
    config.json           # (recommended) focal config renamed as config.json
  app/
    gradio_app.py         # <-- this file

Run:
  pip install -r requirements.txt
  python app/gradio_app.py

Notes:
- This is an educational demo only. Not a medical device.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import librosa
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image


# -----------------------------
# Config + paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_CKPT = MODELS_DIR / "best_model.pth"
DEFAULT_CFG = MODELS_DIR / "config.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Audio + feature helpers
# -----------------------------

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_audio(path: str, sr: int) -> np.ndarray:
    """Load mono audio, resample to sr, peak-normalize."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        return np.zeros(sr, dtype=np.float32)
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)


def pad_or_crop_center(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) == target_len:
        return y
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)), mode="constant")
    start = (len(y) - target_len) // 2
    return y[start : start + target_len]


def split_segments_center(y: np.ndarray, sr: int, num_segments: int, seg_seconds: float) -> np.ndarray:
    seg_len = int(sr * seg_seconds)
    total_len = seg_len * num_segments
    y = pad_or_crop_center(y, total_len)
    return y.reshape(num_segments, seg_len)


def mel_image(
    segment: np.ndarray,
    sr: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    img_size: int,
) -> torch.Tensor:
    """Mel spectrogram -> normalized 3x224x224 tensor (ImageNet norm)."""
    S = librosa.feature.melspectrogram(
        y=segment, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # min-max normalize to [0,1]
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)

    # 1 x n_mels x T  -> resize -> 1 x img x img -> 3 x img x img
    img = torch.from_numpy(S_db).unsqueeze(0).float()
    img = F.interpolate(img.unsqueeze(0), size=(img_size, img_size), mode="bilinear", align_corners=False).squeeze(0)
    img3 = img.repeat(3, 1, 1)

    # ImageNet normalization (important for DenseNet transfer)
    img3 = (img3 - IMAGENET_MEAN) / IMAGENET_STD
    return img3


def make_prob_plot(labels: List[str], probs: np.ndarray) -> Image.Image:
    """Return a bar chart as a PIL image."""
    idx = np.argsort(-probs)
    labels_s = [labels[i] for i in idx]
    probs_s = probs[idx]

    fig = plt.figure(figsize=(7, 4))
    ax = plt.gca()
    ax.bar(labels_s, probs_s)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Probability")
    ax.set_title("Predicted class probabilities")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# -----------------------------
# Model definition (must match training)
# -----------------------------

class DenseNetEncoder(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        # weights loaded from checkpoint state_dict, so weights=None here
        m = torchvision.models.densenet121(weights=None)
        self.features = m.features
        self.out_dim = 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = F.relu(f, inplace=True)
        f = F.adaptive_avg_pool2d(f, (1, 1)).flatten(1)
        return f


class DenseNetBiLSTM(nn.Module):
    def __init__(self, num_classes: int, hidden: int, layers: int, bidirectional: bool, dropout: float):
        super().__init__()
        self.encoder = DenseNetEncoder(pretrained=False)
        self.lstm = nn.LSTM(
            input_size=self.encoder.out_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0 if layers == 1 else dropout,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.encoder(x).view(B, T, -1)
        out, _ = self.lstm(feats)
        last = self.dropout(out[:, -1, :])
        return self.fc(last)


# -----------------------------
# Load checkpoint once (global)
# -----------------------------

@dataclass
class AppConfig:
    sample_rate: int = 22050
    num_segments: int = 5
    segment_seconds: float = 2.0
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    img_size: int = 224
    lstm_hidden: int = 128
    lstm_layers: int = 1
    bidirectional: bool = True
    dropout: float = 0.5


def load_app_config(ckpt: Dict[str, Any]) -> AppConfig:
    cfg = AppConfig()

    if DEFAULT_CFG.exists():
        try:
            d = json.loads(DEFAULT_CFG.read_text(encoding="utf-8"))
            audio = d.get("audio", {})
            mel = d.get("mel", {})
            model = d.get("model", {})
            cfg.sample_rate = int(audio.get("sample_rate", cfg.sample_rate))
            cfg.num_segments = int(audio.get("num_segments", cfg.num_segments))
            cfg.segment_seconds = float(audio.get("segment_seconds", cfg.segment_seconds))
            cfg.n_mels = int(mel.get("n_mels", cfg.n_mels))
            cfg.n_fft = int(mel.get("n_fft", cfg.n_fft))
            cfg.hop_length = int(mel.get("hop_length", cfg.hop_length))
            cfg.img_size = int(mel.get("img_size", cfg.img_size))
            cfg.lstm_hidden = int(model.get("lstm_hidden", cfg.lstm_hidden))
            cfg.lstm_layers = int(model.get("lstm_layers", cfg.lstm_layers))
            cfg.bidirectional = bool(model.get("bidirectional", cfg.bidirectional))
            cfg.dropout = float(model.get("dropout", cfg.dropout))
            return cfg
        except Exception:
            pass

    d2 = ckpt.get("config", {})
    cfg.sample_rate = int(d2.get("sample_rate", cfg.sample_rate))
    cfg.num_segments = int(d2.get("num_segments", cfg.num_segments))
    cfg.segment_seconds = float(d2.get("segment_seconds", cfg.segment_seconds))
    cfg.n_mels = int(d2.get("n_mels", cfg.n_mels))
    cfg.n_fft = int(d2.get("n_fft", cfg.n_fft))
    cfg.hop_length = int(d2.get("hop_length", cfg.hop_length))
    cfg.img_size = int(d2.get("img_size", cfg.img_size))
    cfg.lstm_hidden = int(d2.get("lstm_hidden", cfg.lstm_hidden))
    cfg.lstm_layers = int(d2.get("lstm_layers", cfg.lstm_layers))
    cfg.bidirectional = bool(d2.get("bidirectional", cfg.bidirectional))
    cfg.dropout = float(d2.get("dropout", cfg.dropout))
    return cfg


def load_model() -> Tuple[DenseNetBiLSTM, List[str], Dict[int, str], AppConfig]:
    if not DEFAULT_CKPT.exists():
        raise FileNotFoundError(
            f"Missing model checkpoint: {DEFAULT_CKPT}\n"
            "Tip: copy your focal checkpoint:\n"
            "  cp models/best_model_focal.pth models/best_model.pth\n"
            "  cp models/config_focal.json models/config.json"
        )

    ckpt = torch.load(DEFAULT_CKPT, map_location=DEVICE)
    labels: List[str] = ckpt["labels"]
    id_to_label: Dict[int, str] = ckpt.get("id_to_label") or {i: l for i, l in enumerate(labels)}
    cfg = load_app_config(ckpt)

    model = DenseNetBiLSTM(
        num_classes=len(labels),
        hidden=cfg.lstm_hidden,
        layers=cfg.lstm_layers,
        bidirectional=cfg.bidirectional,
        dropout=cfg.dropout,
    ).to(DEVICE)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, labels, id_to_label, cfg


MODEL, LABELS, ID2LABEL, CFG = load_model()


# -----------------------------
# Inference function
# -----------------------------

@torch.no_grad()
def predict_from_wav(wav_path: str) -> Tuple[str, Image.Image, List[List[Any]]]:
    y = load_audio(wav_path, sr=CFG.sample_rate)
    segs = split_segments_center(y, CFG.sample_rate, CFG.num_segments, CFG.segment_seconds)

    imgs = []
    for s in segs:
        imgs.append(
            mel_image(
                s,
                sr=CFG.sample_rate,
                n_mels=CFG.n_mels,
                n_fft=CFG.n_fft,
                hop_length=CFG.hop_length,
                img_size=CFG.img_size,
            )
        )
    x = torch.stack(imgs, dim=0).unsqueeze(0).to(DEVICE)  # (1,T,3,224,224)

    logits = MODEL(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    pred_id = int(np.argmax(probs))
    pred_label = ID2LABEL.get(pred_id, LABELS[pred_id])

    plot_img = make_prob_plot(LABELS, probs)
    table = [[LABELS[i], float(probs[i])] for i in range(len(LABELS))]
    table.sort(key=lambda r: r[1], reverse=True)

    return pred_label, plot_img, table


def gradio_predict(audio_path: str):
    if audio_path is None or str(audio_path).strip() == "":
        return "Please upload/record a WAV file.", None, None

    try:
        pred_label, plot_img, table = predict_from_wav(audio_path)
        top3 = ", ".join([f"{lbl} ({p:.2f})" for lbl, p in table[:3]])
        pred_text = f"**Prediction:** {pred_label}\n\n**Top‑3:** {top3}\n\n⚠️ Educational demo only — not medical advice."
        return pred_text, plot_img, table
    except Exception as e:
        return f"Error: {e}", None, None


# -----------------------------
# UI
# -----------------------------

def build_ui():
    title = "Lung Sound Classification (6 classes) — DenseNet121 + BiLSTM"
    desc = (
        "Upload or record a lung sound (WAV). The model predicts one of:\n"
        f"- {', '.join(LABELS)}\n\n"
        "⚠️ This is an educational student project demo, **not** a medical device."
    )

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(desc)

        with gr.Row():
            audio = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Lung sound input (WAV)",
            )

        btn = gr.Button("Predict")
        out_text = gr.Markdown()
        out_plot = gr.Image(label="Probabilities (bar chart)")
        out_table = gr.Dataframe(headers=["Label", "Probability"], datatype=["str", "number"], interactive=False)

        btn.click(fn=gradio_predict, inputs=[audio], outputs=[out_text, out_plot, out_table])

        gr.Markdown(
            "### Notes\n"
            "- Best results come from clean recordings (minimal background noise).\n"
            "- If your file is very long/short, the app center-crops/pads to a fixed duration.\n"
            "- If you renamed focal checkpoint to `best_model.pth`, this app uses it by default.\n"
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(show_error=True,share=True)
