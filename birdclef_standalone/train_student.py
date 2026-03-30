from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from birdclef.audio import AudioParams
from birdclef.augmentations import AugmentationConfig, SpectrogramAugmenter
from birdclef.losses import DistillationLoss
from birdclef.models import IMAGE_BACKBONE_NAMES, build_image_classifier, student_model_type
from birdclef.training import evaluate_multilabel, save_checkpoint, train_student_epoch
from birdclef.utils import ensure_dir, make_group_folds, pick_device, seed_everything
from dataset import BirdCLEFWindowDataset, load_manifest, normalize_training_metadata, resolve_class_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BirdCLEF student with pseudo-label distillation.")
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--labeled_window_manifest_csv", required=True)
    parser.add_argument("--pseudo_label_csv", nargs="*", default=[])
    parser.add_argument("--class_list_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--validation_fold", type=int, default=0)
    parser.add_argument("--backbone", choices=IMAGE_BACKBONE_NAMES, default="efficientnet_v2_s")
    parser.add_argument("--use_mil", action="store_true")
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--hard_weight", type=float, default=1.0)
    parser.add_argument("--soft_weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def split_real_windows(metadata: pd.DataFrame, labeled_windows: pd.DataFrame, num_folds: int, validation_fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    file_metadata = metadata.drop_duplicates("soundscape_id").reset_index(drop=True)
    train_idx, val_idx = make_group_folds(file_metadata, n_splits=num_folds)[validation_fold]
    train_ids = set(file_metadata.iloc[train_idx]["soundscape_id"].tolist())
    val_ids = set(file_metadata.iloc[val_idx]["soundscape_id"].tolist())
    train_df = labeled_windows[labeled_windows["soundscape_id"].isin(train_ids)].reset_index(drop=True)
    val_df = labeled_windows[labeled_windows["soundscape_id"].isin(val_ids)].reset_index(drop=True)
    return train_df, val_df


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
    labeled_windows = load_manifest(args.labeled_window_manifest_csv)
    classes = resolve_class_list(labeled_windows, args.class_list_path)
    pseudo_frames = [load_manifest(path) for path in args.pseudo_label_csv]
    pseudo_columns = [class_name for class_name in classes if pseudo_frames and class_name in pseudo_frames[0].columns]

    real_train_df, real_val_df = split_real_windows(
        metadata=metadata,
        labeled_windows=labeled_windows,
        num_folds=args.num_folds,
        validation_fold=args.validation_fold,
    )
    if pseudo_frames:
        pseudo_train_df = pd.concat(pseudo_frames, axis=0, ignore_index=True)
        train_df = pd.concat([real_train_df, pseudo_train_df], axis=0, ignore_index=True)
    else:
        train_df = real_train_df

    augmenter = SpectrogramAugmenter(AugmentationConfig())
    train_ds = BirdCLEFWindowDataset(
        train_df,
        classes=classes,
        params=params,
        training=True,
        augmenter=augmenter,
        pseudo_label_columns=pseudo_columns,
    )
    val_ds = BirdCLEFWindowDataset(real_val_df, classes=classes, params=params, training=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_image_classifier(
        args.backbone,
        num_classes=len(classes),
        pretrained=True,
        use_mil=args.use_mil,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = DistillationLoss(
        hard_weight=args.hard_weight,
        soft_weight=args.soft_weight,
        temperature=args.temperature,
    )

    best_val_loss = float("inf")
    best_payload: dict | None = None
    for epoch in range(args.epochs):
        train_metrics = train_student_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            use_mil=args.use_mil,
            mixup_alpha=args.mixup_alpha,
            distillation_loss=criterion,
        )
        val_metrics, _, _, _, _ = evaluate_multilabel(model, val_loader, device=device, use_mil=args.use_mil)
        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            best_payload = {
                "model_state_dict": model.state_dict(),
                "model_type": student_model_type(args.backbone),
                "classes": classes,
                "use_mil": args.use_mil,
                "audio_params": vars(params),
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "val_loss": val_metrics.loss,
                "pseudo_rounds": args.pseudo_label_csv,
            }
    assert best_payload is not None
    save_checkpoint(output_dir / "student_best.pth", best_payload)


if __name__ == "__main__":
    main()
