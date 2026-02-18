from __future__ import annotations

from io import BytesIO
from typing import List

import numpy as np
import librosa
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from .config import AudioConfig


# ImageNet normalization (DenseNet expects this when using transfer learning)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def mel_image_tensor(segment: np.ndarray, cfg: AudioConfig) -> torch.Tensor:
    """
    Create a DenseNet-ready input tensor from a waveform segment.

    Returns:
      torch.Tensor shape (3, cfg.image_size, cfg.image_size)
      normalized with ImageNet mean/std.
    """
    fmax = cfg.fmax if cfg.fmax is not None else cfg.sample_rate // 2

    S = librosa.feature.melspectrogram(
        y=segment,
        sr=cfg.sample_rate,
        n_mels=cfg.n_mels,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        fmin=cfg.fmin,
        fmax=fmax,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # Normalize to [0,1]
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)

    # (1, n_mels, T)
    img = torch.from_numpy(S_db).unsqueeze(0).float()

    # Resize to (1, H, W) -> H=W=image_size
    img = F.interpolate(
        img.unsqueeze(0),
        size=(cfg.image_size, cfg.image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)  # (1, H, W)

    # Convert to 3 channels
    img3 = img.repeat(3, 1, 1)  # (3, H, W)

    # ImageNet normalization
    img3 = (img3 - IMAGENET_MEAN) / IMAGENET_STD
    return img3


def make_prob_plot(labels: List[str], probs: np.ndarray) -> Image.Image:
    """
    Return a bar chart as a PIL image for Gradio UI.
    probs: shape (num_classes,)
    """
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
