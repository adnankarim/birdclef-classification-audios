from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pandas.errors import EmptyDataError

from birdclef.audio import AudioParams, compute_logmel, iter_window_frames, load_audio_mono, window_spectrogram
from birdclef.models import EfficientNetV2SClassifier
from birdclef.training import load_checkpoint
from birdclef.utils import average_probabilities, file_level_smoothing, load_json, pick_device, temporal_smoothing
from dataset import build_audio_dir_manifest, normalize_inference_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BirdCLEF teacher ensemble inference on Kaggle inputs.")
    parser.add_argument("--teacher_manifest_json", required=True)
    parser.add_argument("--output_csv", default="submission.csv")
    parser.add_argument("--competition_dir", default=None)
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--audio_dir", default=None)
    parser.add_argument("--sample_submission_csv", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--temporal_kernel", type=int, default=3)
    parser.add_argument("--file_smoothing_alpha", type=float, default=0.1)
    parser.add_argument("--max_teachers", type=int, default=None, help="Use only the top-N efficientnet teachers by val_loss.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_sample_schedule(sample_submission_csv: str | Path) -> tuple[pd.DataFrame, dict[str, list[tuple[str, float]]]]:
    sample = pd.read_csv(sample_submission_csv)
    if "row_id" not in sample.columns:
        raise ValueError("sample_submission_csv must contain a row_id column.")

    schedule: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row_id in sample["row_id"].astype(str):
        soundscape_id, separator, end_token = row_id.rpartition("_")
        if not separator:
            raise ValueError(f"Could not parse row_id={row_id!r}. Expected <soundscape_id>_<end_sec>.")
        try:
            end_sec = float(end_token)
        except ValueError as exc:
            raise ValueError(f"Could not parse end time from row_id={row_id!r}.") from exc
        schedule[soundscape_id].append((row_id, end_sec))
    return sample, dict(schedule)


def load_test_manifest(args: argparse.Namespace, target_soundscape_ids: set[str] | None = None) -> pd.DataFrame:
    competition_dir = Path(args.competition_dir) if args.competition_dir else None
    audio_dir = Path(args.audio_dir) if args.audio_dir else None
    if audio_dir is None and competition_dir is not None:
        candidate = competition_dir / "test_soundscapes"
        if candidate.exists():
            audio_dir = candidate

    if args.test_csv:
        try:
            manifest = normalize_inference_manifest(pd.read_csv(args.test_csv), audio_dir=audio_dir)
        except EmptyDataError as exc:
            raise ValueError(
                f"{args.test_csv} is empty. No test audio manifest is available for inference. "
                "Regenerate it from a real test audio directory or pass --audio_dir directly."
            ) from exc
    elif audio_dir is not None:
        manifest = build_audio_dir_manifest(audio_dir)
    else:
        raise ValueError("Provide --competition_dir, --test_csv, or --audio_dir for inference inputs.")

    if manifest.empty:
        raise ValueError("No inference audio files were found.")

    manifest = manifest.drop_duplicates("soundscape_id").reset_index(drop=True)
    if target_soundscape_ids is None:
        return manifest

    manifest = manifest[manifest["soundscape_id"].isin(target_soundscape_ids)].reset_index(drop=True)
    missing_ids = sorted(target_soundscape_ids - set(manifest["soundscape_id"].tolist()))
    if missing_ids:
        raise ValueError(
            "Missing audio files for these sample_submission soundscape_ids: "
            + ", ".join(missing_ids[:10])
            + (" ..." if len(missing_ids) > 10 else "")
        )
    return manifest


def build_default_schedule(soundscape_id: str, num_frames: int, params: AudioParams) -> list[tuple[str, float]]:
    schedule: list[tuple[str, float]] = []
    seen_row_ids: set[str] = set()
    for start_frame in iter_window_frames(num_frames, params):
        end_sec = (start_frame + params.window_num_frames) * params.hop_length / params.sample_rate
        row_id = f"{soundscape_id}_{int(round(end_sec))}"
        if row_id in seen_row_ids:
            continue
        seen_row_ids.add(row_id)
        schedule.append((row_id, end_sec))
    return schedule


def build_window_specs(logmel: np.ndarray | object, params: AudioParams, schedule: list[tuple[str, float]]) -> tuple[list[str], list[np.ndarray]]:
    pad_value = float(logmel.min().item())
    max_start = max(int(logmel.shape[1]) - params.window_num_frames, 0)
    row_ids: list[str] = []
    window_specs: list[np.ndarray] = []
    for row_id, end_sec in schedule:
        end_frame = max(int(round(end_sec * params.sample_rate / params.hop_length)), params.window_num_frames)
        start_frame = min(max(end_frame - params.window_num_frames, 0), max_start)
        spec = window_spectrogram(logmel, params=params, start_frame=start_frame, pad_value=pad_value).numpy()
        row_ids.append(row_id)
        window_specs.append(spec)
    return row_ids, window_specs


def build_model(checkpoint: dict) -> tuple[torch.nn.Module, bool]:
    model_type = checkpoint["model_type"]
    if model_type == "perch_mlp":
        raise ValueError(
            "perch_mlp teacher checkpoints are not supported by infer_teacher_kaggle.py because "
            "they require Perch embeddings at inference time."
        )
    if model_type not in {"efficientnet_v2_s", "efficientnet_v2_s_student"}:
        raise ValueError(f"Unsupported teacher model type: {model_type}")

    model = EfficientNetV2SClassifier(
        num_classes=len(checkpoint["classes"]),
        pretrained=False,
        use_mil=bool(checkpoint.get("use_mil", False)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, bool(checkpoint.get("use_mil", False))


def resolve_checkpoint_path(raw_path: str | Path, manifest_dir: Path) -> Path:
    checkpoint_path = Path(raw_path)
    if checkpoint_path.is_absolute():
        return checkpoint_path
    if checkpoint_path.is_file():
        return checkpoint_path.resolve()
    candidate = (manifest_dir / checkpoint_path).resolve()
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Teacher checkpoint not found: {raw_path}")


def load_teacher_models(
    teacher_manifest_json: str | Path,
    device: torch.device,
    max_teachers: int | None = None,
) -> tuple[list[str], AudioParams, list[tuple[torch.nn.Module, bool]]]:
    teacher_manifest = load_json(teacher_manifest_json)
    classes = teacher_manifest["classes"]
    params = AudioParams(**teacher_manifest["audio_params"])
    manifest_path = Path(teacher_manifest_json).resolve()
    manifest_dir = manifest_path.parent

    teacher_entries: list[dict[str, object]] = []
    for fold in teacher_manifest["folds"]:
        for teacher in fold["teachers"]:
            if teacher["model_type"] == "perch_mlp":
                continue
            teacher_entry = dict(teacher)
            teacher_entry["fold"] = fold["fold"]
            teacher_entries.append(teacher_entry)
    teacher_entries.sort(key=lambda item: (float(item.get("val_loss", float("inf"))), str(item["checkpoint_path"])))
    if max_teachers is not None:
        teacher_entries = teacher_entries[:max_teachers]

    models: list[tuple[torch.nn.Module, bool]] = []
    for teacher in teacher_entries:
        checkpoint_path = resolve_checkpoint_path(teacher["checkpoint_path"], manifest_dir)
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        model, use_mil = build_model(checkpoint)
        model.to(device)
        print(
            f"Loaded teacher fold={teacher['fold']} "
            f"val_loss={float(teacher.get('val_loss', float('nan'))):.6f} "
            f"path={checkpoint_path.name}"
        )
        models.append((model, use_mil))
    if not models:
        raise ValueError("No supported teacher checkpoints were found in teacher_manifest_json.")
    return classes, params, models


@torch.no_grad()
def infer_ensemble_probabilities(
    models: list[tuple[torch.nn.Module, bool]],
    batch: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    inputs = torch.from_numpy(batch).to(device)
    outputs: list[np.ndarray] = []
    for model, use_mil in models:
        model_outputs = model(inputs)
        logits = model_outputs.get("pooled_logits", model_outputs["clip_logits"]) if use_mil else model_outputs["clip_logits"]
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    return average_probabilities(outputs)


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    classes, params, models = load_teacher_models(
        args.teacher_manifest_json,
        device=device,
        max_teachers=args.max_teachers,
    )

    if not args.sample_submission_csv and args.competition_dir:
        candidate = Path(args.competition_dir) / "sample_submission.csv"
        if candidate.exists():
            args.sample_submission_csv = str(candidate)

    sample_submission: pd.DataFrame | None = None
    sample_schedule: dict[str, list[tuple[str, float]]] | None = None
    sample_columns: list[str] | None = None
    if args.sample_submission_csv:
        sample_submission, sample_schedule = load_sample_schedule(args.sample_submission_csv)
        sample_columns = [column for column in sample_submission.columns if column != "row_id"]

    target_soundscape_ids = set(sample_schedule) if sample_schedule is not None else None
    test_manifest = load_test_manifest(args, target_soundscape_ids=target_soundscape_ids)

    rows: list[dict[str, float | str]] = []
    for item in test_manifest.itertuples(index=False):
        waveform = load_audio_mono(item.audio_path, sample_rate=params.sample_rate)
        logmel = compute_logmel(waveform, params)
        schedule = sample_schedule[item.soundscape_id] if sample_schedule is not None else build_default_schedule(
            item.soundscape_id,
            int(logmel.shape[1]),
            params,
        )
        row_ids, window_specs = build_window_specs(logmel, params=params, schedule=schedule)

        probs: list[np.ndarray] = []
        for start in range(0, len(window_specs), args.batch_size):
            batch = np.stack(window_specs[start : start + args.batch_size], axis=0)[:, None, :, :].astype(np.float32)
            preds = infer_ensemble_probabilities(models, batch, device=device)
            probs.append(preds)
        window_probs = np.concatenate(probs, axis=0)
        window_probs = temporal_smoothing(window_probs, kernel_size=args.temporal_kernel)
        window_probs = file_level_smoothing(window_probs, alpha=args.file_smoothing_alpha)

        for row_id, prob in zip(row_ids, window_probs):
            row = {"row_id": row_id}
            row.update({class_name: float(prob[idx]) for idx, class_name in enumerate(classes)})
            rows.append(row)

    submission = pd.DataFrame(rows)
    if sample_submission is not None:
        submission = sample_submission[["row_id"]].merge(submission, on="row_id", how="left")
        if submission["row_id"].duplicated().any():
            raise ValueError("Duplicate row_id values were produced during inference.")
        predicted_columns = [column for column in classes if column in submission.columns]
        if not predicted_columns:
            raise ValueError("None of the teacher output classes are present in sample_submission columns.")
        missing_row_mask = submission[predicted_columns].isna().any(axis=1)
        if missing_row_mask.any():
            missing_row_ids = submission.loc[missing_row_mask, "row_id"].tolist()
            raise ValueError(
                "Inference did not produce predictions for all sample_submission rows: "
                + ", ".join(missing_row_ids[:10])
                + (" ..." if len(missing_row_ids) > 10 else "")
            )
        for column in sample_columns:
            if column not in submission.columns:
                submission[column] = 0.0
        submission = submission[["row_id"] + sample_columns]
    else:
        submission = submission[["row_id"] + classes]
    submission.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
