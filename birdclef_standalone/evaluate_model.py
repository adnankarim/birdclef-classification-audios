from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from birdclef.audio import AudioParams
from birdclef.models import (
    PerchMLPTeacher,
    PerchTemporalStudent,
    build_image_classifier,
    is_supported_image_model_type,
)
from birdclef.training import (
    evaluate_multilabel,
    evaluate_perch_sequence_multilabel,
    load_checkpoint,
)
from birdclef.utils import ensure_dir, make_group_folds, pick_device
from dataset import (
    BirdCLEFPerchSequenceDataset,
    BirdCLEFWindowDataset,
    load_manifest,
    normalize_training_metadata,
    resolve_class_list,
)
from train_perch_student import backfill_perch_embedding_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a BirdCLEF checkpoint and append results to a local log.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--window_manifest_csv", required=True)
    parser.add_argument("--class_list_path", default=None)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--validation_fold", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log_csv", default="outputs/eval_runs/model_eval_log.csv")
    parser.add_argument("--log_jsonl", default="outputs/eval_runs/model_eval_log.jsonl")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--top_n_compare", type=int, default=10)
    parser.add_argument("--run_label", default="")
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


def prepare_validation_windows(args: argparse.Namespace, classes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = normalize_training_metadata(
        pd.read_csv(args.metadata_csv),
        require_audio=False,
        require_labels=False,
    )
    window_manifest = load_manifest(args.window_manifest_csv)
    if args.class_list_path:
        expected_classes = resolve_class_list(window_manifest, args.class_list_path)
        if expected_classes != classes:
            raise ValueError("Checkpoint classes do not match class_list_path.")

    file_metadata = metadata.drop_duplicates("soundscape_id").reset_index(drop=True)
    _, val_idx = make_group_folds(file_metadata, n_splits=args.num_folds)[args.validation_fold]
    val_ids = set(file_metadata.iloc[val_idx]["soundscape_id"].tolist())
    val_df = window_manifest[window_manifest["soundscape_id"].isin(val_ids)].reset_index(drop=True)
    return metadata, val_df


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, object]:
    device = pick_device(args.device)
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    classes = checkpoint["classes"]
    model_type = checkpoint["model_type"]
    _, val_df = prepare_validation_windows(args, classes)

    if model_type == "perch_temporal_student":
        default_embedding_dir = Path(args.window_manifest_csv).resolve().parent / "perch_embeddings"
        val_df = backfill_perch_embedding_paths(
            val_df,
            default_embedding_dir=default_embedding_dir,
        )
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
    else:
        params = AudioParams(**checkpoint["audio_params"])
        use_perch = model_type == "perch_mlp"
        if use_perch:
            default_embedding_dir = Path(args.window_manifest_csv).resolve().parent / "perch_embeddings"
            val_df = backfill_perch_embedding_paths(
                val_df,
                default_embedding_dir=default_embedding_dir,
            )
            model = PerchMLPTeacher(
                embedding_dim=int(checkpoint["embedding_dim"]),
                num_classes=len(classes),
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            dataset = BirdCLEFWindowDataset(
                val_df,
                classes=classes,
                params=params,
                training=False,
                use_perch_embeddings=True,
            )
            use_mil = False
        elif is_supported_image_model_type(model_type):
            model = build_image_classifier(
                model_type,
                num_classes=len(classes),
                pretrained=False,
                use_mil=bool(checkpoint.get("use_mil", False)),
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            dataset = BirdCLEFWindowDataset(
                val_df,
                classes=classes,
                params=params,
                training=False,
            )
            use_mil = bool(checkpoint.get("use_mil", False))
        else:
            raise ValueError(f"Unsupported model_type={model_type!r}")

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        metrics, probs, targets, _, _ = evaluate_multilabel(
            model,
            loader,
            device=device,
            use_mil=use_mil,
            use_perch=use_perch,
        )

    result = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_mtime": datetime.fromtimestamp(checkpoint_path.stat().st_mtime).isoformat(timespec="seconds"),
        "run_label": args.run_label,
        "model_type": model_type,
        "epoch": checkpoint.get("epoch"),
        "train_loss_from_checkpoint": checkpoint.get("train_loss"),
        "val_loss_from_checkpoint": checkpoint.get("val_loss"),
        "validation_fold": args.validation_fold,
        "loss": float(metrics.loss),
        "micro_roc_auc": float(roc_auc_score(targets.ravel(), probs.ravel())),
        "macro_roc_auc": safe_macro_metric(targets, probs, "roc_auc"),
        "macro_ap": safe_macro_metric(targets, probs, "average_precision"),
        "n_windows": int(targets.shape[0]),
        "n_classes": int(targets.shape[1]),
    }
    return result


def append_logs(record: dict[str, object], log_csv: Path, log_jsonl: Path) -> pd.DataFrame:
    ensure_dir(log_csv.parent)
    ensure_dir(log_jsonl.parent)

    row_df = pd.DataFrame([record])
    if log_csv.exists():
        history = pd.read_csv(log_csv)
        history = pd.concat([history, row_df], ignore_index=True)
    else:
        history = row_df
    history.to_csv(log_csv, index=False)

    with log_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return history


def print_comparison(history: pd.DataFrame, top_n: int) -> None:
    columns = [
        "evaluated_at",
        "model_type",
        "checkpoint_name",
        "validation_fold",
        "macro_ap",
        "macro_roc_auc",
        "micro_roc_auc",
        "loss",
    ]
    available_columns = [column for column in columns if column in history.columns]
    ranked = history.sort_values(["macro_ap", "macro_roc_auc", "micro_roc_auc"], ascending=False).head(top_n)
    print("\nTop evaluations:")
    print(ranked[available_columns].to_string(index=False))


def main() -> None:
    args = parse_args()
    record = evaluate_checkpoint(args)
    print(json.dumps(record, indent=2, sort_keys=True))

    log_csv = Path(args.log_csv)
    log_jsonl = Path(args.log_jsonl)
    history = append_logs(record, log_csv=log_csv, log_jsonl=log_jsonl)
    print(f"Wrote log CSV: {log_csv}")
    print(f"Wrote log JSONL: {log_jsonl}")
    print_comparison(history, top_n=args.top_n_compare)

    if args.output_json:
        output_path = Path(args.output_json)
        ensure_dir(output_path.parent)
        output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote report: {output_path}")


if __name__ == "__main__":
    main()
