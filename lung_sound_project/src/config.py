from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


# Project root = .../lung_sound_project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_DIR = PROCESSED_DIR / "manifests"

ICBHI_RAW_DIR = RAW_DIR / "icbhi"
FRAIWAN_RAW_DIR = RAW_DIR / "fraiwan"

MANIFEST_ALL = MANIFEST_DIR / "manifest_all.csv"
MANIFEST_TRAIN = MANIFEST_DIR / "train.csv"
MANIFEST_VAL = MANIFEST_DIR / "val.csv"
MANIFEST_TEST = MANIFEST_DIR / "test.csv"

# Model/artifact paths
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
LABEL_TO_ID_PATH = MODELS_DIR / "label_to_id.json"
ID_TO_LABEL_PATH = MODELS_DIR / "id_to_label.json"
CONFIG_JSON_PATH = MODELS_DIR / "config.json"


@dataclass(frozen=True)
class AudioConfig:
    # Audio standardization
    sample_rate: int = 16000
    clip_seconds: float = 10.0  # total window per sample
    segment_seconds: float = 2.0  # chunk length for LSTM
    # Mel spectrogram
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    fmin: int = 20
    fmax: int | None = None  # None -> sr/2
    # Image for DenseNet
    image_size: int = 224
    # Reproducibility
    seed: int = 42

    @property
    def n_segments(self) -> int:
        # e.g., 10s / 2s = 5 segments
        return int(round(self.clip_seconds / self.segment_seconds))


def ensure_dirs() -> None:
    """Create required output directories if missing."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_config(cfg: AudioConfig, path: Path = CONFIG_JSON_PATH) -> None:
    """Save preprocessing config (must match training + inference)."""
    ensure_dirs()
    payload = asdict(cfg)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_config(path: Path = CONFIG_JSON_PATH) -> AudioConfig:
    """Load preprocessing config (ignore any unknown keys in JSON)."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # keep only keys that exist in AudioConfig
    allowed = set(AudioConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in payload.items() if k in allowed}

    return AudioConfig(**filtered)