from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from birdclef.audio import AudioParams, count_windows, iter_window_frames, window_spectrogram
from birdclef.augmentations import SpectrogramAugmenter
from birdclef.utils import build_class_list, encode_multilabels, normalize_labels

DEFAULT_AUDIO_EXTENSIONS = (".ogg", ".wav", ".flac", ".mp3", ".opus", ".m4a")


def load_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def resolve_class_list(
    metadata: pd.DataFrame,
    class_list_path: str | Path | None = None,
    label_column: str = "labels",
) -> list[str]:
    if class_list_path:
        with open(class_list_path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    return build_class_list(metadata, label_column=label_column)


def _resolve_audio_path(value: object, audio_base_dir: str | Path | None = None) -> str:
    if value is None or pd.isna(value):
        return ""
    raw = str(value).strip()
    if not raw:
        return ""

    candidate = Path(raw)
    if candidate.is_file():
        return str(candidate)

    if audio_base_dir is None:
        return raw

    base_dir = Path(audio_base_dir)
    joined = base_dir / raw
    if joined.is_file():
        return str(joined)
    if joined.suffix:
        return str(joined)
    for ext in DEFAULT_AUDIO_EXTENSIONS:
        ext_candidate = joined.with_suffix(ext)
        if ext_candidate.is_file():
            return str(ext_candidate)
    return str(joined)


def build_audio_dir_manifest(audio_dir: str | Path) -> pd.DataFrame:
    audio_dir = Path(audio_dir)
    rows = []
    for path in sorted(audio_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in DEFAULT_AUDIO_EXTENSIONS:
            continue
        rows.append({"soundscape_id": path.stem, "audio_path": str(path)})
    return pd.DataFrame(rows)


def normalize_training_metadata(
    metadata: pd.DataFrame,
    *,
    audio_column: str = "audio_path",
    id_column: str = "soundscape_id",
    label_column: str = "labels",
    audio_base_dir: str | Path | None = None,
    require_audio: bool = True,
    require_labels: bool = True,
) -> pd.DataFrame:
    normalized = metadata.copy()

    if "soundscape_id" not in normalized.columns:
        for candidate in (id_column, "soundscape_id", "filename", audio_column, "audio_path"):
            if candidate in normalized.columns:
                normalized["soundscape_id"] = normalized[candidate].map(lambda value: Path(str(value)).stem)
                break
    if "soundscape_id" not in normalized.columns:
        raise ValueError("Could not infer soundscape_id. Provide soundscape_id, filename, or audio_path.")

    if "audio_path" in normalized.columns:
        normalized["audio_path"] = normalized["audio_path"].map(
            lambda value: _resolve_audio_path(value, audio_base_dir=audio_base_dir)
        )
    else:
        for candidate in (audio_column, "audio_path", "filename", "soundscape_id"):
            if candidate in normalized.columns:
                normalized["audio_path"] = normalized[candidate].map(
                    lambda value: _resolve_audio_path(value, audio_base_dir=audio_base_dir)
                )
                break
    if require_audio:
        if "audio_path" not in normalized.columns:
            raise ValueError("Could not infer audio_path. Provide audio_path/filename or set --audio_base_dir.")
        missing_audio = normalized["audio_path"].fillna("").astype(str).str.strip().eq("")
        if missing_audio.any():
            raise ValueError("Some metadata rows are missing audio_path values after normalization.")

    if "labels" not in normalized.columns:
        if label_column in normalized.columns and label_column != "labels":
            normalized["labels"] = normalized[label_column].map(
                lambda value: " ".join(normalize_labels(value))
            )
        elif "primary_label" in normalized.columns:
            primary = normalized["primary_label"].map(normalize_labels)
            if "secondary_labels" in normalized.columns:
                secondary = normalized["secondary_labels"].map(normalize_labels)
                normalized["labels"] = [
                    " ".join(sorted(set(primary_labels) | set(secondary_labels)))
                    for primary_labels, secondary_labels in zip(primary, secondary)
                ]
            else:
                normalized["labels"] = primary.map(lambda values: " ".join(values))
    if require_labels:
        if "labels" not in normalized.columns:
            raise ValueError("Could not infer labels. Provide labels or primary_label/secondary_labels.")
        normalized["labels"] = normalized["labels"].map(lambda value: " ".join(normalize_labels(value)))

    return normalized


def normalize_inference_manifest(
    manifest: pd.DataFrame,
    *,
    audio_dir: str | Path | None = None,
    audio_column: str = "audio_path",
    id_column: str = "soundscape_id",
) -> pd.DataFrame:
    normalized = manifest.copy()

    if "soundscape_id" not in normalized.columns:
        for candidate in (id_column, "filename", audio_column, "audio_path"):
            if candidate in normalized.columns:
                normalized["soundscape_id"] = normalized[candidate].map(lambda value: Path(str(value)).stem)
                break
    if "soundscape_id" not in normalized.columns:
        raise ValueError("Inference manifest must include soundscape_id or an audio filename/path column.")

    if "audio_path" in normalized.columns:
        normalized["audio_path"] = normalized["audio_path"].map(
            lambda value: _resolve_audio_path(value, audio_base_dir=audio_dir)
        )
    else:
        source_column = None
        for candidate in (audio_column, "filename"):
            if candidate in normalized.columns:
                source_column = candidate
                break
        if source_column is not None:
            normalized["audio_path"] = normalized[source_column].map(
                lambda value: _resolve_audio_path(value, audio_base_dir=audio_dir)
            )
        elif audio_dir is not None:
            normalized["audio_path"] = normalized["soundscape_id"].map(
                lambda value: _resolve_audio_path(value, audio_base_dir=audio_dir)
            )

    if "audio_path" not in normalized.columns:
        raise ValueError("Could not infer audio_path for inference. Provide audio_path or --audio_dir.")
    missing_audio = normalized["audio_path"].fillna("").astype(str).str.strip().eq("")
    if missing_audio.any():
        raise ValueError("Some inference rows are missing audio_path values after normalization.")

    return normalized


def build_window_manifest(
    metadata: pd.DataFrame,
    file_manifest: pd.DataFrame,
    params: AudioParams,
    label_column: str = "labels",
    strong_start_column: str | None = None,
    strong_end_column: str | None = None,
) -> pd.DataFrame:
    manifest = metadata.merge(
        file_manifest[["soundscape_id", "spec_path", "perch_embedding_path", "num_frames", "duration_sec"]],
        on="soundscape_id",
        how="inner",
    )
    rows: list[dict[str, Any]] = []
    for soundscape_id, group_df in manifest.groupby("soundscape_id", sort=False):
        base_row = group_df.iloc[0]
        weak_labels = set()
        for raw in group_df[label_column].fillna(""):
            weak_labels.update(normalize_labels(raw))
        events = []
        if strong_start_column and strong_end_column:
            for event_row in group_df.itertuples(index=False):
                start_value = getattr(event_row, strong_start_column, None)
                end_value = getattr(event_row, strong_end_column, None)
                if start_value is None or end_value is None or pd.isna(start_value) or pd.isna(end_value):
                    continue
                events.append(
                    {
                        "start_sec": float(start_value),
                        "end_sec": float(end_value),
                        "labels": normalize_labels(getattr(event_row, label_column, "")),
                    }
                )

        for window_idx, start_frame in enumerate(iter_window_frames(int(base_row.num_frames), params)):
            start_sec = start_frame * params.hop_length / params.sample_rate
            end_sec = start_sec + params.window_seconds
            labels: set[str]
            if events:
                labels = set()
                for event in events:
                    overlap = max(0.0, min(end_sec, event["end_sec"]) - max(start_sec, event["start_sec"]))
                    if overlap > 0.0:
                        labels.update(event["labels"])
            else:
                labels = set(weak_labels)
            rows.append(
                {
                    "soundscape_id": soundscape_id,
                    "spec_path": base_row.spec_path,
                    "perch_embedding_path": base_row.get("perch_embedding_path", ""),
                    "window_idx": window_idx,
                    "start_frame": start_frame,
                    "end_frame": start_frame + params.window_num_frames,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "labels": " ".join(sorted(labels)),
                }
            )
    return pd.DataFrame(rows)


class BirdCLEFWindowDataset(Dataset):
    def __init__(
        self,
        window_manifest: pd.DataFrame,
        classes: list[str],
        params: AudioParams,
        training: bool = False,
        augmenter: SpectrogramAugmenter | None = None,
        noise_bank: list[torch.Tensor] | None = None,
        use_perch_embeddings: bool = False,
        pseudo_label_columns: list[str] | None = None,
    ):
        self.window_manifest = window_manifest.reset_index(drop=True)
        self.classes = classes
        self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        self.params = params
        self.training = training
        self.augmenter = augmenter
        self.noise_bank = noise_bank or []
        self.use_perch_embeddings = use_perch_embeddings
        self.pseudo_label_columns = pseudo_label_columns or []

    @staticmethod
    @lru_cache(maxsize=2048)
    def _load_spec(path: str) -> torch.Tensor:
        array = np.load(path)
        return torch.tensor(array, dtype=torch.float32)

    @staticmethod
    @lru_cache(maxsize=2048)
    def _load_embedding(path: str) -> torch.Tensor:
        array = np.load(path)
        return torch.tensor(array, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.window_manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.window_manifest.iloc[index]
        spec = self._load_spec(str(row.spec_path))
        window = window_spectrogram(
            spec,
            params=self.params,
            start_frame=int(row.start_frame),
            pad_value=float(spec.min().item()),
        )
        window = window.unsqueeze(0)
        if self.training and self.augmenter is not None:
            window = self.augmenter.apply(window, noise_bank=self.noise_bank)

        hard_targets = torch.tensor(
            encode_multilabels(normalize_labels(row.labels), self.class_to_idx),
            dtype=torch.float32,
        )
        sample: dict[str, Any] = {
            "inputs": window,
            "hard_targets": hard_targets,
            "window_idx": int(row.window_idx),
            "soundscape_id": row.soundscape_id,
            "row_id": f"{row.soundscape_id}_{int(round(float(row.end_sec)))}",
        }
        if self.pseudo_label_columns:
            soft_targets = row[self.pseudo_label_columns].to_numpy(dtype=np.float32, copy=True)
            hard_targets_np = hard_targets.numpy()
            if np.isnan(soft_targets).any():
                soft_targets = np.where(np.isnan(soft_targets), hard_targets_np, soft_targets)
            soft_targets = np.clip(soft_targets, 0.0, 1.0)
            sample["soft_targets"] = torch.tensor(soft_targets, dtype=torch.float32)
        if self.use_perch_embeddings and isinstance(row.get("perch_embedding_path", ""), str) and row.perch_embedding_path:
            embeddings = self._load_embedding(str(row.perch_embedding_path))
            sample["perch_embedding"] = embeddings[int(row.window_idx)]
        return sample


def save_class_list(classes: list[str], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for label in classes:
            handle.write(f"{label}\n")


def save_window_manifest(window_manifest: pd.DataFrame, path: str | Path) -> None:
    window_manifest.to_csv(path, index=False)
