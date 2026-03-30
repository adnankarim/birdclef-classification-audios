from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from birdclef.audio import AudioParams
from birdclef.augmentations import AugmentationConfig, SpectrogramAugmenter
from birdclef.models import IMAGE_BACKBONE_NAMES, PerchMLPTeacher, build_image_classifier
from birdclef.training import (
    evaluate_multilabel,
    save_checkpoint,
    train_multilabel_epoch,
    train_perch_epoch,
)
from birdclef.utils import ensure_dir, make_group_folds, pick_device, save_json, seed_everything
from dataset import BirdCLEFWindowDataset, load_manifest, normalize_training_metadata, resolve_class_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BirdCLEF teacher ensemble.")
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--window_manifest_csv", required=True)
    parser.add_argument("--class_list_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--num_effnet_teachers", type=int, default=2)
    parser.add_argument(
        "--teacher_model_types",
        nargs="*",
        default=None,
        choices=IMAGE_BACKBONE_NAMES,
        help="Optional explicit list of image backbones to train per fold. Example: --teacher_model_types efficientnet_v2_m convnext_small",
    )
    parser.add_argument("--train_perch_teacher", action="store_true")
    parser.add_argument("--use_mil", action="store_true")
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--noise_window_manifest_csv", default=None)
    parser.add_argument("--max_noise_windows", type=int, default=128)
    return parser.parse_args()


def build_noise_bank(
    noise_window_manifest_csv: str | None,
    classes: list[str],
    params: AudioParams,
    max_noise_windows: int,
) -> list[torch.Tensor]:
    if not noise_window_manifest_csv:
        return []
    noise_manifest = load_manifest(noise_window_manifest_csv)
    if noise_manifest.empty:
        return []
    noise_ds = BirdCLEFWindowDataset(
        noise_manifest.head(max_noise_windows),
        classes=classes,
        params=params,
        training=False,
    )
    return [noise_ds[idx]["inputs"] for idx in range(len(noise_ds))]


def train_image_teacher(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    classes: list[str],
    params: AudioParams,
    args: argparse.Namespace,
    device: torch.device,
    output_path: Path,
    noise_bank: list[torch.Tensor],
    model_type: str,
) -> dict:
    augmenter = SpectrogramAugmenter(AugmentationConfig())
    train_ds = BirdCLEFWindowDataset(
        train_df,
        classes=classes,
        params=params,
        training=True,
        augmenter=augmenter,
        noise_bank=noise_bank,
    )
    val_ds = BirdCLEFWindowDataset(val_df, classes=classes, params=params, training=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_image_classifier(
        model_type,
        num_classes=len(classes),
        pretrained=True,
        use_mil=args.use_mil,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_payload: dict | None = None
    for epoch in range(args.epochs):
        train_metrics = train_multilabel_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            use_mil=args.use_mil,
            mixup_alpha=args.mixup_alpha,
        )
        val_metrics, _, _, _, _ = evaluate_multilabel(model, val_loader, device=device, use_mil=args.use_mil)
        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            best_payload = {
                "model_state_dict": model.state_dict(),
                "model_type": model_type,
                "classes": classes,
                "use_mil": args.use_mil,
                "audio_params": vars(params),
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "val_loss": val_metrics.loss,
            }
    assert best_payload is not None
    save_checkpoint(output_path, best_payload)
    return {"checkpoint_path": str(output_path), "val_loss": best_val_loss, "model_type": model_type}


def train_perch_teacher(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    classes: list[str],
    params: AudioParams,
    args: argparse.Namespace,
    device: torch.device,
    output_path: Path,
) -> dict:
    sample_path = next((path for path in train_df["perch_embedding_path"].tolist() if isinstance(path, str) and path), None)
    if sample_path is None:
        raise ValueError("Perch teacher requested, but perch_embedding_path is missing from the manifest.")
    embedding_dim = int(np.load(sample_path).shape[-1])

    train_ds = BirdCLEFWindowDataset(
        train_df,
        classes=classes,
        params=params,
        training=False,
        use_perch_embeddings=True,
    )
    val_ds = BirdCLEFWindowDataset(
        val_df,
        classes=classes,
        params=params,
        training=False,
        use_perch_embeddings=True,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = PerchMLPTeacher(embedding_dim=embedding_dim, num_classes=len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val_loss = float("inf")
    best_payload: dict | None = None
    for epoch in range(args.epochs):
        train_metrics = train_perch_epoch(model, train_loader, optimizer, device=device)
        val_metrics, _, _, _, _ = evaluate_multilabel(model, val_loader, device=device, use_mil=False, use_perch=True)
        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            best_payload = {
                "model_state_dict": model.state_dict(),
                "model_type": "perch_mlp",
                "classes": classes,
                "embedding_dim": embedding_dim,
                "audio_params": vars(params),
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "val_loss": val_metrics.loss,
            }
    assert best_payload is not None
    save_checkpoint(output_path, best_payload)
    return {"checkpoint_path": str(output_path), "val_loss": best_val_loss, "model_type": "perch_mlp"}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    params = AudioParams()
    output_dir = ensure_dir(args.output_dir)

    metadata = normalize_training_metadata(
        pd.read_csv(args.metadata_csv),
        require_audio=False,
        require_labels=False,
    )
    window_manifest = load_manifest(args.window_manifest_csv)
    classes = resolve_class_list(window_manifest, args.class_list_path)
    noise_bank = build_noise_bank(args.noise_window_manifest_csv, classes, params, args.max_noise_windows)

    file_metadata = metadata.drop_duplicates("soundscape_id").reset_index(drop=True)
    folds = make_group_folds(file_metadata, n_splits=args.num_folds)

    teacher_manifest: dict[str, object] = {
        "classes": classes,
        "audio_params": vars(params),
        "folds": [],
    }
    teacher_model_types = args.teacher_model_types or ["efficientnet_v2_s"] * args.num_effnet_teachers
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_ids = set(file_metadata.iloc[train_idx]["soundscape_id"].tolist())
        val_ids = set(file_metadata.iloc[val_idx]["soundscape_id"].tolist())
        train_df = window_manifest[window_manifest["soundscape_id"].isin(train_ids)].reset_index(drop=True)
        val_df = window_manifest[window_manifest["soundscape_id"].isin(val_ids)].reset_index(drop=True)
        fold_dir = ensure_dir(output_dir / f"fold_{fold_idx}")
        fold_entry = {"fold": fold_idx, "teachers": []}

        for teacher_idx, model_type in enumerate(teacher_model_types):
            checkpoint_path = fold_dir / f"{model_type}_teacher_{teacher_idx}.pth"
            result = train_image_teacher(
                train_df,
                val_df,
                classes,
                params,
                args,
                device,
                checkpoint_path,
                noise_bank,
                model_type,
            )
            fold_entry["teachers"].append(result)

        if args.train_perch_teacher:
            checkpoint_path = fold_dir / "perch_teacher.pth"
            result = train_perch_teacher(train_df, val_df, classes, params, args, device, checkpoint_path)
            fold_entry["teachers"].append(result)

        teacher_manifest["folds"].append(fold_entry)

    save_json(teacher_manifest, output_dir / "teacher_manifest.json")


if __name__ == "__main__":
    main()
