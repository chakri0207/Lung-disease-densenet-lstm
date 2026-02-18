from __future__ import annotations

from pathlib import Path
import numpy as np
import librosa

from .config import AudioConfig


def load_audio(path: str | Path, sr: int) -> np.ndarray:
    """
    Load mono audio at target sample rate and peak-normalize.
    Returns float32 waveform.
    """
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    if y.size == 0:
        return np.zeros(sr, dtype=np.float32)

    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)


def pad_or_crop_center(y: np.ndarray, target_len: int) -> np.ndarray:
    """Center crop if too long, right pad with zeros if too short."""
    if len(y) == target_len:
        return y
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)), mode="constant")

    start = (len(y) - target_len) // 2
    return y[start : start + target_len]


def standardize_clip(y: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """Force audio to fixed clip length: cfg.clip_seconds."""
    target_len = int(cfg.sample_rate * cfg.clip_seconds)
    return pad_or_crop_center(y, target_len)


def split_into_segments(y: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """
    Split standardized audio into cfg.n_segments segments.
    Output shape: (n_segments, seg_len)
    """
    seg_len = int(cfg.sample_rate * cfg.segment_seconds)
    total_len = seg_len * cfg.n_segments
    y = pad_or_crop_center(y, total_len)
    return y.reshape(cfg.n_segments, seg_len)


def load_and_segment(audio_path: str | Path, cfg: AudioConfig) -> np.ndarray:
    """Load -> standardize -> split into segments."""
    y = load_audio(audio_path, sr=cfg.sample_rate)
    y = standardize_clip(y, cfg)
    return split_into_segments(y, cfg)
