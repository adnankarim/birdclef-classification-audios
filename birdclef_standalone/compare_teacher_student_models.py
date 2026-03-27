from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from birdclef.audio import AudioParams, compute_logmel, load_audio_mono
from birdclef.models import EfficientNetV2SClassifier, PerchMLPTeacher
from birdclef.training import load_checkpoint, predict_probabilities
from birdclef.utils import average_probabilities, encode_multilabels, normalize_labels, pick_device
from dataset import BirdCLEFWindowDataset, load_manifest, normalize_training_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare teacher ensemble, FP32 ONNX, and INT8 ONNX on the same labeled file sample."
    )
    parser.add_argument("--teacher_manifest_json", required=True)
    parser.add_argument("--window_manifest_csv", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--class_list_path", required=True)
    parser.add_argument("--fp32_model_path", default=None)
    parser.add_argument("--int8_model_path", default=None)
    parser.add_argument("--audio_base_dir", default=None)
    parser.add_argument("--sample_n_files", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--aggregate", choices=["max", "mean"], default="max")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_json", default="teacher_student_compare_report.json")
    parser.add_argument("--per_file_csv", default="teacher_student_compare_per_file.csv")
    return parser.parse_args()


def load_classes(class_list_path: str | Path) -> list[str]:
    return [line.strip() for line in Path(class_list_path).read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_targets(metadata: pd.DataFrame, class_to_idx: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for soundscape_id, group in metadata.groupby("soundscape_id", sort=False):
        labels: set[str] = set()
        audio_path = None
        for row in group.itertuples(index=False):
            labels.update(normalize_labels(getattr(row, "labels", "")))
            candidate_audio = getattr(row, "audio_path", "")
            if not audio_path and isinstance(candidate_audio, str) and candidate_audio:
                audio_path = candidate_audio
        rows.append(
            {
                "soundscape_id": soundscape_id,
                "audio_path": audio_path,
                "labels": " ".join(sorted(labels)),
                "target": encode_multilabels(sorted(labels), class_to_idx),
            }
        )
    return pd.DataFrame(rows)


def build_model(checkpoint: dict) -> tuple[torch.nn.Module, bool, bool]:
    classes = checkpoint["classes"]
    model_type = checkpoint["model_type"]
    if model_type in {"efficientnet_v2_s", "efficientnet_v2_s_student"}:
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


def aggregate_probabilities(soundscape_ids: list[str], probabilities: np.ndarray, aggregate: str) -> dict[str, np.ndarray]:
    buckets: dict[str, list[np.ndarray]] = {}
    for soundscape_id, prob in zip(soundscape_ids, probabilities):
        buckets.setdefault(soundscape_id, []).append(prob)
    aggregated: dict[str, np.ndarray] = {}
    for soundscape_id, probs in buckets.items():
        stacked = np.stack(probs, axis=0)
        if aggregate == "mean":
            aggregated[soundscape_id] = stacked.mean(axis=0)
        else:
            aggregated[soundscape_id] = stacked.max(axis=0)
    return aggregated


def build_windows(logmel: np.ndarray | object, params: AudioParams) -> np.ndarray:
    pad_value = float(logmel.min().item())
    max_start = max(int(logmel.shape[1]) - params.window_num_frames, 0)
    windows: list[np.ndarray] = []
    start = 0
    while True:
        end_frame = start + params.window_num_frames
        if end_frame <= logmel.shape[1]:
            window = logmel[:, start:end_frame]
        else:
            import torch.nn.functional as F

            window = F.pad(logmel[:, start:], (0, end_frame - logmel.shape[1]), value=pad_value)
        windows.append(window.numpy())
        if start >= max_start:
            break
        start = min(start + params.stride_num_frames, max_start)
    return np.stack(windows, axis=0)[:, None, :, :].astype(np.float32)


def infer_onnx_file(session: ort.InferenceSession, audio_path: Path, params: AudioParams, batch_size: int, aggregate: str) -> np.ndarray:
    waveform = load_audio_mono(audio_path, sample_rate=params.sample_rate)
    logmel = compute_logmel(waveform, params)
    windows = build_windows(logmel, params=params)
    outputs: list[np.ndarray] = []
    for start in range(0, len(windows), batch_size):
        batch = windows[start : start + batch_size]
        preds = session.run(["probabilities"], {"inputs": batch})[0]
        outputs.append(preds)
    probs = np.concatenate(outputs, axis=0)
    if aggregate == "mean":
        return probs.mean(axis=0)
    return probs.max(axis=0)


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


def metric_block(y_true: np.ndarray, y_score: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_micro_roc_auc": float(roc_auc_score(y_true.ravel(), y_score.ravel())),
        f"{prefix}_macro_roc_auc": safe_macro_metric(y_true, y_score, "roc_auc"),
        f"{prefix}_macro_ap": safe_macro_metric(y_true, y_score, "average_precision"),
    }


def delta_block(metrics: dict[str, float], left: str, right: str) -> dict[str, float]:
    return {
        f"delta_{left}_vs_{right}_micro_roc_auc": metrics[f"{left}_micro_roc_auc"] - metrics[f"{right}_micro_roc_auc"],
        f"delta_{left}_vs_{right}_macro_roc_auc": metrics[f"{left}_macro_roc_auc"] - metrics[f"{right}_macro_roc_auc"],
        f"delta_{left}_vs_{right}_macro_ap": metrics[f"{left}_macro_ap"] - metrics[f"{right}_macro_ap"],
    }


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    classes = load_classes(args.class_list_path)
    class_to_idx = {label: idx for idx, label in enumerate(classes)}

    metadata = normalize_training_metadata(
        pd.read_csv(args.metadata_csv),
        audio_base_dir=args.audio_base_dir,
        require_audio=True,
        require_labels=False,
    )
    file_targets = resolve_targets(metadata, class_to_idx)
    window_manifest = load_manifest(args.window_manifest_csv)
    file_targets = file_targets[file_targets["soundscape_id"].isin(window_manifest["soundscape_id"])].reset_index(drop=True)
    if file_targets.empty:
        raise ValueError("No overlapping soundscape_id values between metadata_csv and window_manifest_csv.")

    sample_n = min(args.sample_n_files, len(file_targets))
    file_targets = file_targets.sample(n=sample_n, random_state=args.seed).sort_values("soundscape_id").reset_index(drop=True)
    sampled_ids = set(file_targets["soundscape_id"])

    teacher_manifest = json.loads(Path(args.teacher_manifest_json).read_text(encoding="utf-8"))
    teacher_classes = teacher_manifest["classes"]
    if teacher_classes != classes:
        raise ValueError("Teacher manifest classes do not match class_list_path.")
    params = AudioParams(**teacher_manifest["audio_params"])

    sampled_windows = window_manifest[window_manifest["soundscape_id"].isin(sampled_ids)].reset_index(drop=True)
    has_perch_embeddings = sampled_windows.get("perch_embedding_path", pd.Series(dtype=str)).fillna("").ne("").any()
    teacher_dataset = BirdCLEFWindowDataset(
        sampled_windows,
        classes=classes,
        params=params,
        training=False,
        use_perch_embeddings=has_perch_embeddings,
    )
    teacher_loader = DataLoader(
        teacher_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    teacher_outputs: list[np.ndarray] = []
    teacher_soundscape_ids: list[str] | None = None
    for fold in teacher_manifest["folds"]:
        for teacher in fold["teachers"]:
            checkpoint = load_checkpoint(teacher["checkpoint_path"], map_location="cpu")
            model, use_mil, use_perch = build_model(checkpoint)
            model.to(device)
            predictions = predict_probabilities(model, teacher_loader, device=device, use_mil=use_mil, use_perch=use_perch)
            teacher_outputs.append(predictions["probabilities"])
            teacher_soundscape_ids = predictions["soundscape_id"]

    assert teacher_soundscape_ids is not None
    teacher_window_probs = average_probabilities(teacher_outputs)
    teacher_by_file = aggregate_probabilities(teacher_soundscape_ids, teacher_window_probs, aggregate=args.aggregate)

    fp32_session = ort.InferenceSession(str(args.fp32_model_path), providers=["CPUExecutionProvider"]) if args.fp32_model_path else None
    int8_session = ort.InferenceSession(str(args.int8_model_path), providers=["CPUExecutionProvider"]) if args.int8_model_path else None

    y_true: list[np.ndarray] = []
    teacher_probs: list[np.ndarray] = []
    fp32_probs: list[np.ndarray] = []
    int8_probs: list[np.ndarray] = []
    per_file_rows: list[dict[str, float | str | int]] = []

    for row in file_targets.itertuples(index=False):
        soundscape_id = row.soundscape_id
        audio_path = Path(str(row.audio_path))
        target = row.target
        teacher_pred = teacher_by_file[soundscape_id]

        y_true.append(target)
        teacher_probs.append(teacher_pred)

        per_file_row: dict[str, float | str | int] = {
            "soundscape_id": soundscape_id,
            "audio_path": str(audio_path),
            "n_labels": int(np.asarray(target).sum()),
            "teacher_top1_prob": float(teacher_pred.max()),
        }

        if fp32_session is not None:
            fp32_pred = infer_onnx_file(fp32_session, audio_path, params=params, batch_size=args.batch_size, aggregate=args.aggregate)
            fp32_probs.append(fp32_pred)
            per_file_row["fp32_top1_prob"] = float(fp32_pred.max())
            per_file_row["teacher_vs_fp32_mean_abs_diff"] = float(np.abs(teacher_pred - fp32_pred).mean())
            per_file_row["teacher_vs_fp32_max_abs_diff"] = float(np.abs(teacher_pred - fp32_pred).max())

        if int8_session is not None:
            int8_pred = infer_onnx_file(int8_session, audio_path, params=params, batch_size=args.batch_size, aggregate=args.aggregate)
            int8_probs.append(int8_pred)
            per_file_row["int8_top1_prob"] = float(int8_pred.max())
            per_file_row["teacher_vs_int8_mean_abs_diff"] = float(np.abs(teacher_pred - int8_pred).mean())
            per_file_row["teacher_vs_int8_max_abs_diff"] = float(np.abs(teacher_pred - int8_pred).max())

        if fp32_session is not None and int8_session is not None:
            per_file_row["fp32_vs_int8_mean_abs_diff"] = float(np.abs(fp32_pred - int8_pred).mean())
            per_file_row["fp32_vs_int8_max_abs_diff"] = float(np.abs(fp32_pred - int8_pred).max())

        per_file_rows.append(per_file_row)

    y_true_np = np.stack(y_true, axis=0)
    teacher_np = np.stack(teacher_probs, axis=0)

    report: dict[str, object] = {
        "n_files": int(len(file_targets)),
        "aggregate": args.aggregate,
        "metrics": {},
    }
    metrics: dict[str, float] = {}
    metrics.update(metric_block(y_true_np, teacher_np, "teacher"))

    if fp32_probs:
        fp32_np = np.stack(fp32_probs, axis=0)
        metrics.update(metric_block(y_true_np, fp32_np, "fp32"))
        metrics.update(delta_block(metrics, "fp32", "teacher"))
        metrics["teacher_vs_fp32_mean_abs_diff"] = float(np.abs(teacher_np - fp32_np).mean())
        metrics["teacher_vs_fp32_median_abs_diff"] = float(np.median(np.abs(teacher_np - fp32_np)))
        metrics["teacher_vs_fp32_max_abs_diff"] = float(np.abs(teacher_np - fp32_np).max())

    if int8_probs:
        int8_np = np.stack(int8_probs, axis=0)
        metrics.update(metric_block(y_true_np, int8_np, "int8"))
        metrics.update(delta_block(metrics, "int8", "teacher"))
        metrics["teacher_vs_int8_mean_abs_diff"] = float(np.abs(teacher_np - int8_np).mean())
        metrics["teacher_vs_int8_median_abs_diff"] = float(np.median(np.abs(teacher_np - int8_np)))
        metrics["teacher_vs_int8_max_abs_diff"] = float(np.abs(teacher_np - int8_np).max())
        if fp32_probs:
            metrics.update(delta_block(metrics, "int8", "fp32"))
            metrics["fp32_vs_int8_mean_abs_diff"] = float(np.abs(fp32_np - int8_np).mean())
            metrics["fp32_vs_int8_median_abs_diff"] = float(np.median(np.abs(fp32_np - int8_np)))
            metrics["fp32_vs_int8_max_abs_diff"] = float(np.abs(fp32_np - int8_np).max())

    report["metrics"] = metrics

    per_file_df = pd.DataFrame(per_file_rows)
    sort_column = "teacher_vs_int8_mean_abs_diff" if "teacher_vs_int8_mean_abs_diff" in per_file_df.columns else (
        "teacher_vs_fp32_mean_abs_diff" if "teacher_vs_fp32_mean_abs_diff" in per_file_df.columns else "teacher_top1_prob"
    )
    per_file_df = per_file_df.sort_values(sort_column, ascending=False)

    Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    per_file_df.to_csv(args.per_file_csv, index=False)

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote report: {args.output_json}")
    print(f"Wrote per-file CSV: {args.per_file_csv}")


if __name__ == "__main__":
    main()
