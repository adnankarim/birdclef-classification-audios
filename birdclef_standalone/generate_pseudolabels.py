from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from birdclef.audio import AudioParams
from birdclef.models import EfficientNetV2SClassifier, PerchMLPTeacher
from birdclef.training import load_checkpoint, predict_probabilities
from birdclef.utils import average_probabilities, ensure_dir, load_json, pick_device, save_json
from dataset import BirdCLEFWindowDataset, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate soft pseudo-labels from a teacher ensemble.")
    parser.add_argument("--teacher_manifest_json", required=True)
    parser.add_argument("--window_manifest_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--extra_checkpoint", nargs="*", default=[])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--keep_threshold", type=float, default=0.7)
    parser.add_argument("--label_threshold", type=float, default=0.5)
    parser.add_argument("--round_idx", type=int, default=1)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def build_model(checkpoint: dict) -> tuple[torch.nn.Module, bool, bool]:
    classes = checkpoint["classes"]
    model_type = checkpoint["model_type"]
    if model_type == "efficientnet_v2_s":
        model = EfficientNetV2SClassifier(
            num_classes=len(classes),
            pretrained=False,
            use_mil=bool(checkpoint.get("use_mil", False)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, bool(checkpoint.get("use_mil", False)), False
    if model_type == "efficientnet_v2_s_student":
        model = EfficientNetV2SClassifier(
            num_classes=len(classes),
            pretrained=False,
            use_mil=bool(checkpoint.get("use_mil", False)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, bool(checkpoint.get("use_mil", False)), False
    if model_type == "perch_mlp":
        model = PerchMLPTeacher(
            embedding_dim=int(checkpoint["embedding_dim"]),
            num_classes=len(classes),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, False, True
    raise ValueError(f"Unsupported teacher model type: {model_type}")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    teacher_manifest = load_json(args.teacher_manifest_json)
    classes = teacher_manifest["classes"]
    params = AudioParams(**teacher_manifest["audio_params"])
    window_manifest = load_manifest(args.window_manifest_csv)
    has_perch_embeddings = window_manifest.get("perch_embedding_path", pd.Series(dtype=str)).fillna("").ne("").any()
    needs_perch = any(
        teacher["model_type"] == "perch_mlp"
        for fold in teacher_manifest["folds"]
        for teacher in fold["teachers"]
    )
    if needs_perch and not has_perch_embeddings:
        raise ValueError("Perch teacher checkpoints require perch_embedding_path values in the window manifest.")

    dataset = BirdCLEFWindowDataset(
        window_manifest,
        classes=classes,
        params=params,
        training=False,
        use_perch_embeddings=has_perch_embeddings,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    ensemble_outputs: list[np.ndarray] = []
    last_meta: dict | None = None
    for fold in teacher_manifest["folds"]:
        for teacher in fold["teachers"]:
            checkpoint = load_checkpoint(teacher["checkpoint_path"], map_location="cpu")
            model, use_mil, use_perch = build_model(checkpoint)
            model.to(device)
            predictions = predict_probabilities(model, loader, device=device, use_mil=use_mil, use_perch=use_perch)
            ensemble_outputs.append(predictions["probabilities"])
            last_meta = predictions

    for checkpoint_path in args.extra_checkpoint:
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        model, use_mil, use_perch = build_model(checkpoint)
        model.to(device)
        predictions = predict_probabilities(model, loader, device=device, use_mil=use_mil, use_perch=use_perch)
        ensemble_outputs.append(predictions["probabilities"])
        last_meta = predictions

    if last_meta is None:
        raise ValueError("No teacher checkpoints found in teacher manifest.")

    ensemble_probs = average_probabilities(ensemble_outputs)
    confidences = ensemble_probs.max(axis=1)
    keep_mask = confidences >= args.keep_threshold

    kept = window_manifest.loc[keep_mask].reset_index(drop=True).copy()
    kept["confidence"] = confidences[keep_mask]
    kept["labels"] = [
        " ".join([classes[idx] for idx, prob in enumerate(row) if prob >= args.label_threshold])
        for row in ensemble_probs[keep_mask]
    ]
    prob_df = pd.DataFrame(ensemble_probs[keep_mask], columns=classes, index=kept.index)
    kept = pd.concat([kept, prob_df], axis=1)

    kept.to_csv(output_dir / f"pseudo_labels_round{args.round_idx}.csv", index=False)
    save_json(
        {
            "round_idx": args.round_idx,
            "num_candidates": int(len(window_manifest)),
            "num_kept": int(keep_mask.sum()),
            "keep_threshold": args.keep_threshold,
            "label_threshold": args.label_threshold,
        },
        output_dir / f"pseudo_labels_round{args.round_idx}.json",
    )


if __name__ == "__main__":
    main()
