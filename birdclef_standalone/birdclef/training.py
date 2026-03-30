from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from birdclef.augmentations import apply_mixup
from birdclef.losses import AsymmetricLossMultiLabel, DistillationLoss


@dataclass
class EpochMetrics:
    loss: float
    macro_ap: float | None = None


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def train_multilabel_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_mil: bool = False,
    mixup_alpha: float = 0.0,
) -> EpochMetrics:
    model.train()
    criterion = AsymmetricLossMultiLabel()
    losses: list[float] = []
    for batch in tqdm(loader, leave=False):
        batch = move_batch_to_device(batch, device)
        inputs = batch["inputs"]
        targets = batch["hard_targets"]
        inputs, targets, _ = apply_mixup(inputs, targets, alpha=mixup_alpha)
        outputs = model(inputs)
        logits = outputs.get("pooled_logits", outputs["clip_logits"]) if use_mil else outputs["clip_logits"]
        loss = criterion(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return EpochMetrics(loss=float(np.mean(losses)))


def train_perch_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochMetrics:
    model.train()
    criterion = AsymmetricLossMultiLabel()
    losses: list[float] = []
    for batch in tqdm(loader, leave=False):
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["perch_embedding"])
        loss = criterion(outputs["clip_logits"], batch["hard_targets"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return EpochMetrics(loss=float(np.mean(losses)))


def train_perch_sequence_student_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    distillation_loss: DistillationLoss | None = None,
) -> EpochMetrics:
    model.train()
    criterion = distillation_loss or DistillationLoss()
    losses: list[float] = []
    for batch in tqdm(loader, leave=False):
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["perch_embeddings"], valid_mask=batch["valid_mask"])
        logits = outputs["frame_logits"]
        valid_mask = batch["valid_mask"].bool()
        if not valid_mask.any():
            continue
        hard_targets = batch["hard_targets"][valid_mask]
        soft_targets = batch.get("soft_targets")
        if soft_targets is not None:
            soft_targets = soft_targets[valid_mask]
        loss = criterion(logits[valid_mask], hard_targets, teacher_probs=soft_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return EpochMetrics(loss=float(np.mean(losses)) if losses else float("nan"))


@torch.no_grad()
def evaluate_multilabel(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_mil: bool = False,
    use_perch: bool = False,
) -> tuple[EpochMetrics, np.ndarray, np.ndarray, list[str], list[int]]:
    model.eval()
    criterion = AsymmetricLossMultiLabel()
    losses: list[float] = []
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    soundscape_ids: list[str] = []
    window_indices: list[int] = []
    for batch in tqdm(loader, leave=False):
        batch = move_batch_to_device(batch, device)
        if use_perch:
            outputs = model(batch["perch_embedding"])
        else:
            outputs = model(batch["inputs"])
        logits = outputs.get("pooled_logits", outputs["clip_logits"]) if use_mil else outputs["clip_logits"]
        loss = criterion(logits, batch["hard_targets"])
        losses.append(float(loss.detach().cpu().item()))
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(batch["hard_targets"].cpu().numpy())
        soundscape_ids.extend(batch["soundscape_id"])
        window_indices.extend(batch["window_idx"])
    return (
        EpochMetrics(loss=float(np.mean(losses))),
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_targets, axis=0),
        soundscape_ids,
        window_indices,
    )


@torch.no_grad()
def evaluate_perch_sequence_multilabel(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[EpochMetrics, np.ndarray, np.ndarray, list[str], list[int]]:
    model.eval()
    criterion = DistillationLoss(hard_weight=1.0, soft_weight=0.0)
    losses: list[float] = []
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    soundscape_ids: list[str] = []
    window_indices: list[int] = []
    for batch in tqdm(loader, leave=False):
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["perch_embeddings"], valid_mask=batch["valid_mask"])
        logits = outputs["frame_logits"]
        valid_mask = batch["valid_mask"].bool()
        if not valid_mask.any():
            continue
        valid_logits = logits[valid_mask]
        valid_targets = batch["hard_targets"][valid_mask]
        loss = criterion(valid_logits, valid_targets, teacher_probs=None)
        losses.append(float(loss.detach().cpu().item()))
        all_probs.append(torch.sigmoid(valid_logits).cpu().numpy())
        all_targets.append(valid_targets.cpu().numpy())
        for row_idx, soundscape_id in enumerate(batch["soundscape_id"]):
            valid_indices = valid_mask[row_idx].nonzero(as_tuple=False).flatten().tolist()
            soundscape_ids.extend([soundscape_id] * len(valid_indices))
            window_indices.extend(valid_indices)
    empty = np.zeros((0, 0), dtype=np.float32)
    return (
        EpochMetrics(loss=float(np.mean(losses)) if losses else float("nan")),
        np.concatenate(all_probs, axis=0) if all_probs else empty,
        np.concatenate(all_targets, axis=0) if all_targets else empty,
        soundscape_ids,
        window_indices,
    )


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_mil: bool = False,
    use_perch: bool = False,
) -> dict[str, Any]:
    model.eval()
    probabilities: list[np.ndarray] = []
    soundscape_ids: list[str] = []
    window_indices: list[int] = []
    row_ids: list[str] = []
    for batch in tqdm(loader, leave=False):
        batch = move_batch_to_device(batch, device)
        if use_perch:
            outputs = model(batch["perch_embedding"])
            logits = outputs["clip_logits"]
        else:
            outputs = model(batch["inputs"])
            logits = outputs.get("pooled_logits", outputs["clip_logits"]) if use_mil else outputs["clip_logits"]
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        soundscape_ids.extend(batch["soundscape_id"])
        window_indices.extend(batch["window_idx"])
        row_ids.extend(batch["row_id"])
    return {
        "probabilities": np.concatenate(probabilities, axis=0),
        "soundscape_id": soundscape_ids,
        "window_idx": window_indices,
        "row_id": row_ids,
    }


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def train_student_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_mil: bool = False,
    mixup_alpha: float = 0.0,
    distillation_loss: DistillationLoss | None = None,
) -> EpochMetrics:
    model.train()
    criterion = distillation_loss or DistillationLoss()
    losses: list[float] = []
    for batch in tqdm(loader, leave=False):
        batch = move_batch_to_device(batch, device)
        inputs = batch["inputs"]
        hard_targets = batch["hard_targets"]
        soft_targets = batch.get("soft_targets")
        inputs, hard_targets, soft_targets = apply_mixup(inputs, hard_targets, soft_targets, alpha=mixup_alpha)
        outputs = model(inputs)
        logits = outputs.get("pooled_logits", outputs["clip_logits"]) if use_mil else outputs["clip_logits"]
        loss = criterion(logits, hard_targets, teacher_probs=soft_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return EpochMetrics(loss=float(np.mean(losses)))
