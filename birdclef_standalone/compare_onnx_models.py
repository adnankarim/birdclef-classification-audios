from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from birdclef.audio import AudioParams, compute_logmel, load_audio_mono
from birdclef.utils import encode_multilabels, normalize_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FP32 and INT8 ONNX BirdCLEF models on the same labeled audio sample."
    )
    parser.add_argument("--fp32_model_path", required=True)
    parser.add_argument("--int8_model_path", required=True)
    parser.add_argument("--class_list_path", required=True)
    parser.add_argument("--metadata_csv", required=True, help="CSV with audio_path and labels columns.")
    parser.add_argument("--audio_base_dir", default=None, help="Optional base dir to resolve relative audio paths.")
    parser.add_argument("--sample_n_files", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--aggregate", choices=["max", "mean"], default="max")
    parser.add_argument("--output_json", default="onnx_compare_report.json")
    parser.add_argument("--per_file_csv", default="onnx_compare_per_file.csv")
    return parser.parse_args()


def resolve_audio_path(raw_value: object, audio_base_dir: str | Path | None) -> Path:
    raw_text = str(raw_value).strip()
    candidate = Path(raw_text)
    if candidate.is_file():
        return candidate
    if audio_base_dir is None:
        return candidate
    joined = Path(audio_base_dir) / raw_text
    return joined


def load_classes(class_list_path: str | Path) -> list[str]:
    return [line.strip() for line in Path(class_list_path).read_text(encoding="utf-8").splitlines() if line.strip()]


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


def infer_file(session: ort.InferenceSession, audio_path: Path, params: AudioParams, batch_size: int, aggregate: str) -> np.ndarray:
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


def main() -> None:
    args = parse_args()
    classes = load_classes(args.class_list_path)
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    params = AudioParams()

    metadata = pd.read_csv(args.metadata_csv)
    if "audio_path" not in metadata.columns or "labels" not in metadata.columns:
        raise ValueError("metadata_csv must contain audio_path and labels columns.")
    metadata = metadata.copy()
    metadata["resolved_audio_path"] = metadata["audio_path"].map(lambda value: str(resolve_audio_path(value, args.audio_base_dir)))
    metadata = metadata[metadata["resolved_audio_path"].map(lambda value: Path(value).is_file())].reset_index(drop=True)
    if metadata.empty:
        raise ValueError("No readable audio files were found in metadata_csv.")

    sample_n = min(args.sample_n_files, len(metadata))
    metadata = metadata.sample(n=sample_n, random_state=args.seed).reset_index(drop=True)

    fp32_session = ort.InferenceSession(str(args.fp32_model_path), providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(str(args.int8_model_path), providers=["CPUExecutionProvider"])

    y_true: list[np.ndarray] = []
    fp32_probs: list[np.ndarray] = []
    int8_probs: list[np.ndarray] = []
    per_file_rows: list[dict[str, float | str | int]] = []

    for row in metadata.itertuples(index=False):
        audio_path = Path(row.resolved_audio_path)
        target = encode_multilabels(normalize_labels(row.labels), class_to_idx)
        fp32_pred = infer_file(fp32_session, audio_path, params=params, batch_size=args.batch_size, aggregate=args.aggregate)
        int8_pred = infer_file(int8_session, audio_path, params=params, batch_size=args.batch_size, aggregate=args.aggregate)
        drift = np.abs(fp32_pred - int8_pred)

        y_true.append(target)
        fp32_probs.append(fp32_pred)
        int8_probs.append(int8_pred)
        per_file_rows.append(
            {
                "soundscape_id": getattr(row, "soundscape_id", audio_path.stem),
                "audio_path": str(audio_path),
                "n_labels": int(target.sum()),
                "fp32_top1_prob": float(fp32_pred.max()),
                "int8_top1_prob": float(int8_pred.max()),
                "mean_abs_diff": float(drift.mean()),
                "max_abs_diff": float(drift.max()),
            }
        )

    y_true_np = np.stack(y_true, axis=0)
    fp32_np = np.stack(fp32_probs, axis=0)
    int8_np = np.stack(int8_probs, axis=0)
    drift = np.abs(fp32_np - int8_np)

    report = {
        "n_files": int(len(metadata)),
        "aggregate": args.aggregate,
        "metrics": {
            "fp32_micro_roc_auc": float(roc_auc_score(y_true_np.ravel(), fp32_np.ravel())),
            "int8_micro_roc_auc": float(roc_auc_score(y_true_np.ravel(), int8_np.ravel())),
            "fp32_macro_roc_auc": safe_macro_metric(y_true_np, fp32_np, "roc_auc"),
            "int8_macro_roc_auc": safe_macro_metric(y_true_np, int8_np, "roc_auc"),
            "fp32_macro_ap": safe_macro_metric(y_true_np, fp32_np, "average_precision"),
            "int8_macro_ap": safe_macro_metric(y_true_np, int8_np, "average_precision"),
            "mean_abs_diff": float(drift.mean()),
            "median_abs_diff": float(np.median(drift)),
            "max_abs_diff": float(drift.max()),
        },
    }
    report["metrics"]["delta_micro_roc_auc"] = report["metrics"]["int8_micro_roc_auc"] - report["metrics"]["fp32_micro_roc_auc"]
    report["metrics"]["delta_macro_roc_auc"] = report["metrics"]["int8_macro_roc_auc"] - report["metrics"]["fp32_macro_roc_auc"]
    report["metrics"]["delta_macro_ap"] = report["metrics"]["int8_macro_ap"] - report["metrics"]["fp32_macro_ap"]

    per_file_df = pd.DataFrame(per_file_rows).sort_values("mean_abs_diff", ascending=False)
    Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    per_file_df.to_csv(args.per_file_csv, index=False)

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote report: {args.output_json}")
    print(f"Wrote per-file drift CSV: {args.per_file_csv}")


if __name__ == "__main__":
    main()
