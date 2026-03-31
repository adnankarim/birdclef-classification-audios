from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from birdclef.models import PerchTemporalStudent
from birdclef.training import evaluate_perch_sequence_multilabel, load_checkpoint
from birdclef.utils import make_group_folds, pick_device
from dataset import BirdCLEFPerchSequenceDataset, load_manifest, normalize_training_metadata, resolve_class_list
from train_perch_student import backfill_perch_embedding_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Perch temporal student on its held-out validation fold.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--window_manifest_csv", required=True)
    parser.add_argument("--class_list_path", default=None)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--validation_fold", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def safe_macro_metric(y_true: np.ndarray, y_score: np.ndarray, metric_name: str) -> float:
    valid = y_true.sum(axis=0) > 0
    valid &= (y_true.shape[0] - y_true.sum(axis=0)) > 0
    if not valid.any():
        return float("nan")
    if metric_name == "roc_auc":
        return float(roc_auc_score(y_true[:, valid], y_score[:, valid], average="macro"))
    if metric_name == "average_precision":
        return float(average_precision_score(y_true[:, valid], y_score[:, valid], average="macro"))
    raise ValueError(metric_name)


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    checkpoint = load_checkpoint(args.checkpoint_path, map_location="cpu")
    classes = checkpoint["classes"]
    if args.class_list_path:
        expected_classes = resolve_class_list(load_manifest(args.window_manifest_csv), args.class_list_path)
        if expected_classes != classes:
            raise ValueError("checkpoint classes do not match class_list_path.")

    metadata = normalize_training_metadata(
        pd.read_csv(args.metadata_csv),
        require_audio=False,
        require_labels=False,
    )
    window_manifest = load_manifest(args.window_manifest_csv)
    default_embedding_dir = Path(args.window_manifest_csv).resolve().parent / "perch_embeddings"
    window_manifest = backfill_perch_embedding_paths(
        window_manifest,
        default_embedding_dir=default_embedding_dir,
    )

    file_metadata = metadata.drop_duplicates("soundscape_id").reset_index(drop=True)
    train_idx, val_idx = make_group_folds(file_metadata, n_splits=args.num_folds)[args.validation_fold]
    val_ids = set(file_metadata.iloc[val_idx]["soundscape_id"].tolist())
    val_df = window_manifest[window_manifest["soundscape_id"].isin(val_ids)].reset_index(drop=True)

    dataset = BirdCLEFPerchSequenceDataset(
        val_df,
        classes=classes,
        max_windows=int(checkpoint["max_windows"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = PerchTemporalStudent(
        embedding_dim=int(checkpoint["embedding_dim"]),
        num_classes=len(classes),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_heads=int(checkpoint["num_heads"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
        max_positions=int(checkpoint["max_windows"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics, probs, targets, _, _ = evaluate_perch_sequence_multilabel(model, loader, device=device)
    result = {
        "loss": float(metrics.loss),
        "micro_roc_auc": float(roc_auc_score(targets.ravel(), probs.ravel())),
        "macro_roc_auc": safe_macro_metric(targets, probs, "roc_auc"),
        "macro_ap": safe_macro_metric(targets, probs, "average_precision"),
        "n_windows": int(targets.shape[0]),
        "n_classes": int(targets.shape[1]),
        "validation_fold": args.validation_fold,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote report: {args.output_json}")


if __name__ == "__main__":
    main()
