from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

from .config import ID_TO_LABEL_PATH, LABEL_TO_ID_PATH


def build_from_labels(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create label_to_id and id_to_label from a labels list."""
    label_to_id = {l: i for i, l in enumerate(labels)}
    id_to_label = {i: l for i, l in enumerate(labels)}
    return label_to_id, id_to_label


def load_json_map(path: Path) -> Dict[Any, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_label_maps(
    *,
    labels_fallback: List[str] | None = None,
    id_to_label_path: Path = ID_TO_LABEL_PATH,
    label_to_id_path: Path = LABEL_TO_ID_PATH,
) -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    """
    Load label maps from models/*.json if present.
    Fallback to `labels_fallback` (e.g., ckpt["labels"]) if files missing.
    Returns: (label_to_id, id_to_label, labels_list)
    """
    id_to_label_raw = load_json_map(id_to_label_path)
    label_to_id_raw = load_json_map(label_to_id_path)

    # normalize id_to_label keys to int
    id_to_label: Dict[int, str] = {}
    for k, v in id_to_label_raw.items():
        try:
            id_to_label[int(k)] = str(v)
        except Exception:
            pass

    label_to_id: Dict[str, int] = {}
    for k, v in label_to_id_raw.items():
        try:
            label_to_id[str(k)] = int(v)
        except Exception:
            pass

    if id_to_label and label_to_id:
        # derive labels list from id_to_label
        labels = [id_to_label[i] for i in sorted(id_to_label.keys())]
        return label_to_id, id_to_label, labels

    # fallback to provided labels list (from checkpoint)
    if labels_fallback is not None and len(labels_fallback) > 0:
        label_to_id, id_to_label = build_from_labels(labels_fallback)
        return label_to_id, id_to_label, labels_fallback

    # last resort (empty)
    return {}, {}, []
