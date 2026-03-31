from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from birdclef.losses import DistillationLoss
from birdclef.models import PerchTemporalStudent
from birdclef.training import (
    evaluate_perch_sequence_multilabel,
    save_checkpoint,
    train_perch_sequence_student_epoch,
)
from birdclef.utils import ensure_dir, make_group_folds, pick_device, seed_everything
from dataset import BirdCLEFPerchSequenceDataset, load_manifest, normalize_training_metadata, resolve_class_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Perch sequence student on cached window embeddings.")
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--labeled_window_manifest_csv", required=True)
    parser.add_argument("--pseudo_label_csv", nargs="*", default=[])
    parser.add_argument("--class_list_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--validation_fold", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--hard_weight", type=float, default=1.0)
    parser.add_argument("--soft_weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def split_real_windows(
    metadata: pd.DataFrame,
    labeled_windows: pd.DataFrame,
    num_folds: int,
    validation_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    file_metadata = metadata.drop_duplicates("soundscape_id").reset_index(drop=True)
    train_idx, val_idx = make_group_folds(file_metadata, n_splits=num_folds)[validation_fold]
    train_ids = set(file_metadata.iloc[train_idx]["soundscape_id"].tolist())
    val_ids = set(file_metadata.iloc[val_idx]["soundscape_id"].tolist())
    train_df = labeled_windows[labeled_windows["soundscape_id"].isin(train_ids)].reset_index(drop=True)
    val_df = labeled_windows[labeled_windows["soundscape_id"].isin(val_ids)].reset_index(drop=True)
    return train_df, val_df


def resolve_embedding_dim(*frames: pd.DataFrame) -> int:
    for frame in frames:
        if "perch_embedding_path" not in frame.columns:
            continue
        for path in frame["perch_embedding_path"].fillna("").astype(str):
            if path:
                import numpy as np

                array = np.load(path)
                if array.ndim != 2:
                    raise ValueError(f"Expected 2D Perch embedding array at {path}, got shape {array.shape}")
                return int(array.shape[-1])
    raise ValueError(
        "No perch_embedding_path values were found. Run preprocess.py with --compute_perch_embeddings first."
    )


def backfill_perch_embedding_paths(
    frame: pd.DataFrame,
    *,
    default_embedding_dir: Path,
    reference_windows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    if "perch_embedding_path" not in result.columns:
        result["perch_embedding_path"] = ""
    result["perch_embedding_path"] = result["perch_embedding_path"].fillna("").astype(str)

    if reference_windows is not None and {"soundscape_id", "window_idx", "perch_embedding_path"}.issubset(reference_windows.columns):
        lookup = (
            reference_windows[["soundscape_id", "window_idx", "perch_embedding_path"]]
            .drop_duplicates(["soundscape_id", "window_idx"])
            .rename(columns={"perch_embedding_path": "_perch_embedding_path_ref"})
        )
        result = result.merge(lookup, on=["soundscape_id", "window_idx"], how="left")
        mask = result["perch_embedding_path"].eq("") & result["_perch_embedding_path_ref"].fillna("").astype(str).ne("")
        result.loc[mask, "perch_embedding_path"] = result.loc[mask, "_perch_embedding_path_ref"].astype(str)
        result = result.drop(columns=["_perch_embedding_path_ref"])

    def infer_path(soundscape_id: object, current_path: object) -> str:
        raw = str(current_path).strip()
        if raw:
            return raw
        candidate = default_embedding_dir / f"{soundscape_id}.npy"
        if candidate.is_file():
            return str(candidate)
        return ""

    result["perch_embedding_path"] = [
        infer_path(soundscape_id, perch_path)
        for soundscape_id, perch_path in zip(result["soundscape_id"], result["perch_embedding_path"])
    ]
    return result


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    metadata = normalize_training_metadata(
        pd.read_csv(args.metadata_csv),
        require_audio=False,
        require_labels=False,
    )
    labeled_windows = load_manifest(args.labeled_window_manifest_csv)
    default_embedding_dir = Path(args.labeled_window_manifest_csv).resolve().parent / "perch_embeddings"
    labeled_windows = backfill_perch_embedding_paths(
        labeled_windows,
        default_embedding_dir=default_embedding_dir,
    )
    classes = resolve_class_list(labeled_windows, args.class_list_path)
    pseudo_frames = [
        backfill_perch_embedding_paths(
            load_manifest(path),
            default_embedding_dir=default_embedding_dir,
            reference_windows=labeled_windows,
        )
        for path in args.pseudo_label_csv
    ]
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

    embedding_dim = resolve_embedding_dim(train_df, real_val_df)
    train_ds = BirdCLEFPerchSequenceDataset(
        train_df,
        classes=classes,
        pseudo_label_columns=pseudo_columns,
        max_windows=args.max_windows,
    )
    val_ds = BirdCLEFPerchSequenceDataset(
        real_val_df,
        classes=classes,
        max_windows=train_ds.max_windows,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = PerchTemporalStudent(
        embedding_dim=embedding_dim,
        num_classes=len(classes),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_positions=train_ds.max_windows,
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
        train_metrics = train_perch_sequence_student_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            distillation_loss=criterion,
        )
        val_metrics, _, _, _, _ = evaluate_perch_sequence_multilabel(
            model,
            val_loader,
            device=device,
        )
        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            best_payload = {
                "model_state_dict": model.state_dict(),
                "model_type": "perch_temporal_student",
                "classes": classes,
                "embedding_dim": embedding_dim,
                "hidden_dim": args.hidden_dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "max_windows": train_ds.max_windows,
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "val_loss": val_metrics.loss,
                "pseudo_rounds": args.pseudo_label_csv,
            }
    assert best_payload is not None
    save_checkpoint(output_dir / "perch_student_best.pth", best_payload)


if __name__ == "__main__":
    main()
