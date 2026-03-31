from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from birdclef.models import PerchTemporalStudent
from birdclef.training import load_checkpoint, predict_perch_sequence_probabilities
from birdclef.utils import ensure_dir, save_json
from dataset import BirdCLEFPerchSequenceDataset, load_manifest, resolve_class_list
from train_perch_student import backfill_perch_embedding_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pseudo-label CSVs from a Perch temporal student checkpoint.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--window_manifest_csv", required=True)
    parser.add_argument("--class_list_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--keep_threshold", type=float, default=0.7)
    parser.add_argument("--label_threshold", type=float, default=0.5)
    parser.add_argument("--round_idx", type=int, default=1)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    checkpoint = load_checkpoint(args.checkpoint_path, map_location="cpu")
    if checkpoint["model_type"] != "perch_temporal_student":
        raise ValueError(f"Expected perch_temporal_student checkpoint, got {checkpoint['model_type']!r}.")
    classes = checkpoint["classes"]

    window_manifest = load_manifest(args.window_manifest_csv)
    if args.class_list_path:
        expected_classes = resolve_class_list(window_manifest, args.class_list_path)
        if expected_classes != classes:
            raise ValueError("Checkpoint classes do not match class_list_path.")

    default_embedding_dir = Path(args.window_manifest_csv).resolve().parent / "perch_embeddings"
    window_manifest = backfill_perch_embedding_paths(
        window_manifest,
        default_embedding_dir=default_embedding_dir,
    )

    dataset = BirdCLEFPerchSequenceDataset(
        window_manifest,
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

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    predictions = predict_perch_sequence_probabilities(model, loader, device=device)
    pred_df = pd.DataFrame(predictions["probabilities"], columns=classes)
    pred_df.insert(0, "window_idx", predictions["window_idx"])
    pred_df.insert(0, "soundscape_id", predictions["soundscape_id"])

    merged = window_manifest.merge(pred_df, on=["soundscape_id", "window_idx"], how="inner")
    confidences = merged[classes].max(axis=1).to_numpy(dtype=np.float32)
    keep_mask = confidences >= args.keep_threshold

    kept = merged.loc[keep_mask].reset_index(drop=True).copy()
    kept["confidence"] = confidences[keep_mask]
    kept["labels"] = [
        " ".join([classes[idx] for idx, prob in enumerate(row) if prob >= args.label_threshold])
        for row in kept[classes].to_numpy(dtype=np.float32, copy=True)
    ]

    csv_path = output_dir / f"pseudo_labels_round{args.round_idx}.csv"
    json_path = output_dir / f"pseudo_labels_round{args.round_idx}.json"
    kept.to_csv(csv_path, index=False)
    save_json(
        {
            "round_idx": args.round_idx,
            "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
            "num_candidates": int(len(window_manifest)),
            "num_predicted": int(len(merged)),
            "num_kept": int(len(kept)),
            "keep_threshold": args.keep_threshold,
            "label_threshold": args.label_threshold,
        },
        json_path,
    )
    print(json.dumps(json.loads(json_path.read_text(encoding="utf-8")), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
