from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(payload: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def normalize_labels(value: str | float | list[str]) -> list[str]:
    if isinstance(value, list):
        return sorted({str(item).strip() for item in value if str(item).strip()})
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, list):
                return normalize_labels(parsed)
        except json.JSONDecodeError:
            pass
    text = text.replace(",", " ")
    return sorted({token.strip() for token in text.split() if token.strip()})


def build_class_list(metadata: pd.DataFrame, label_column: str = "labels") -> list[str]:
    species = set()
    for raw in metadata[label_column].fillna(""):
        species.update(normalize_labels(raw))
    return sorted(species)


def encode_multilabels(labels: Iterable[str], class_to_idx: dict[str, int]) -> np.ndarray:
    target = np.zeros(len(class_to_idx), dtype=np.float32)
    for label in labels:
        if label in class_to_idx:
            target[class_to_idx[label]] = 1.0
    return target


def build_group_ids(metadata: pd.DataFrame) -> np.ndarray:
    parts = []
    for column in ("author", "recordist", "site"):
        if column in metadata.columns:
            parts.append(metadata[column].fillna("unknown").astype(str))
    if not parts:
        return np.arange(len(metadata))
    return parts[0].str.cat(parts[1:], sep="|").to_numpy()


def make_group_folds(metadata: pd.DataFrame, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = GroupKFold(n_splits=n_splits)
    groups = build_group_ids(metadata)
    dummy_x = np.zeros((len(metadata), 1), dtype=np.float32)
    dummy_y = np.zeros(len(metadata), dtype=np.int32)
    return list(splitter.split(dummy_x, dummy_y, groups=groups))


def average_probabilities(probability_sets: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(probability_sets, axis=0)
    return stacked.mean(axis=0)


def temporal_smoothing(window_probs: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if kernel_size <= 1 or window_probs.shape[0] <= 1:
        return window_probs
    pad = kernel_size // 2
    padded = np.pad(window_probs, ((pad, pad), (0, 0)), mode="edge")
    smoothed = []
    for idx in range(window_probs.shape[0]):
        smoothed.append(padded[idx : idx + kernel_size].mean(axis=0))
    return np.stack(smoothed, axis=0)


def file_level_smoothing(window_probs: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    file_mean = window_probs.mean(axis=0, keepdims=True)
    return (1.0 - alpha) * window_probs + alpha * file_mean


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

