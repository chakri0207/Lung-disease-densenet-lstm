from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .config import (
    AudioConfig,
    BEST_MODEL_PATH,
    CONFIG_JSON_PATH,
    load_config,
)
from .features import mel_image_tensor, make_prob_plot
from .label_map import load_label_maps
from .model import DenseNetBiLSTM


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PredictionResult:
    pred_label: str
    probs: np.ndarray
    table: List[List[Any]]
    plot_img: Any


class Predictor:
    def __init__(
        self,
        model: DenseNetBiLSTM,
        cfg: AudioConfig,
        labels: List[str],
        id_to_label: Dict[int, str],
        device: torch.device,
    ):
        self.model = model
        self.cfg = cfg
        self.labels = labels
        self.id_to_label = id_to_label
        self.device = device

    @torch.no_grad()
    def predict_file(self, audio_path: str | Path) -> Tuple[str, Any, List[List[Any]]]:
        """
        Multi-window voting inference:
          - start window
          - center window
          - end window

        Each window is standardized to cfg.clip_seconds, then split into cfg.n_segments.
        Probabilities are averaged across windows.
        """

        from .audio_preprocess import load_audio, pad_or_crop_center

        # Load full audio (mono, resampled, peak-normalized)
        y = load_audio(audio_path, sr=self.cfg.sample_rate)

        clip_len = int(self.cfg.sample_rate * self.cfg.clip_seconds)
        seg_len = int(self.cfg.sample_rate * self.cfg.segment_seconds)

        T = self.cfg.n_segments  # derived from clip_seconds/segment_seconds
        total_len = seg_len * T

        # Build 3 windows: start / center / end
        if len(y) < clip_len:
            y_full = pad_or_crop_center(y, clip_len)
            windows = [y_full, y_full, y_full]
        else:
            start_w = y[:clip_len]
            c0 = (len(y) - clip_len) // 2
            center_w = y[c0 : c0 + clip_len]
            end_w = y[-clip_len:]
            windows = [start_w, center_w, end_w]

        probs_sum = None

        # Predict each window, average probs
        for w in windows:
            w = pad_or_crop_center(w, total_len)
            segs = w.reshape(T, seg_len)  # (T, seg_len)

            imgs = [mel_image_tensor(seg, self.cfg) for seg in segs]
            x = torch.stack(imgs, dim=0).unsqueeze(0).to(self.device)  # (1, T, 3, H, W)

            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

            probs_sum = probs if probs_sum is None else (probs_sum + probs)

        probs_avg = probs_sum / len(windows)

        pred_id = int(np.argmax(probs_avg))
        pred_label = self.id_to_label.get(pred_id, self.labels[pred_id])

        plot_img = make_prob_plot(self.labels, probs_avg)
        table = [[self.labels[i], float(probs_avg[i])] for i in range(len(self.labels))]
        table.sort(key=lambda r: r[1], reverse=True)

        return pred_label, plot_img, table


def load_predictor(
    ckpt_path: Path = BEST_MODEL_PATH,
    cfg_path: Path = CONFIG_JSON_PATH,
) -> Predictor:
    """
    Loads predictor from:
      - models/config.json (base)
      - models/best_model.pth (checkpoint)
    Then overrides cfg with ckpt["config"] where keys match AudioConfig.
    """

    device = _device()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json: {cfg_path}")

    # Base cfg from JSON (your load_config ignores unknown keys like 'labels')
    cfg = load_config(cfg_path)

    # Load checkpoint
    ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location=device)

    # ---- Override cfg with checkpoint config (source of truth) ----
    ckpt_cfg = ckpt.get("config", {}) or {}
    allowed = set(AudioConfig.__dataclass_fields__.keys())
    override = {k: v for k, v in ckpt_cfg.items() if k in allowed}
    if override:
        cfg = AudioConfig(**{**cfg.__dict__, **override})

    # ---- Labels + mappings ----
    labels_fallback = ckpt.get("labels")
    if isinstance(labels_fallback, list):
        labels_fallback = [str(x) for x in labels_fallback]
    else:
        labels_fallback = None

    _, id_to_label, labels = load_label_maps(labels_fallback=labels_fallback)

    if not labels:
        raise RuntimeError(
            "No labels found. Provide either:\n"
            "- ckpt['labels'] in checkpoint, OR\n"
            "- models/id_to_label.json + models/label_to_id.json"
        )

    # ---- Model hyperparams (prefer ckpt_cfg if present) ----
    lstm_hidden = int(ckpt_cfg.get("lstm_hidden", 128))
    lstm_layers = int(ckpt_cfg.get("lstm_layers", 1))
    bidirectional = bool(ckpt_cfg.get("bidirectional", True))
    dropout = float(ckpt_cfg.get("dropout", 0.5))

    model = DenseNetBiLSTM(
        num_classes=len(labels),
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        bidirectional=bidirectional,
        dropout=dropout,
    ).to(device)

    # Load weights
    state_dict = ckpt.get("state_dict", ckpt)  # allow raw state_dict too
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return Predictor(model=model, cfg=cfg, labels=labels, id_to_label=id_to_label, device=device)
