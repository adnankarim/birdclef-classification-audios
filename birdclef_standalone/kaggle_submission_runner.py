from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BirdCLEF Kaggle submission inference with hidden-test/dry-run fallback."
    )
    parser.add_argument(
        "--artifact_dir",
        default="/kaggle/input/datasets/adnankarem/dataclef",
        help="Mounted Kaggle dataset containing model artifacts.",
    )
    parser.add_argument(
        "--competition_dir",
        default="/kaggle/input/competitions/birdclef-2026",
        help="Mounted BirdCLEF competition directory.",
    )
    parser.add_argument(
        "--wheel_path",
        default=None,
        help="Optional onnxruntime wheel path for offline installation.",
    )
    parser.add_argument(
        "--dryrun_n_files",
        type=int,
        default=5,
        help="How many train_audio files to use when hidden test audio is absent.",
    )
    parser.add_argument(
        "--working_dir",
        default="/kaggle/working",
        help="Writable directory for dry-run manifest/predictions and submission output.",
    )
    return parser.parse_args()


def ensure_onnxruntime(wheel_path: str | None) -> None:
    try:
        import onnxruntime  # noqa: F401
        return
    except ModuleNotFoundError:
        if not wheel_path:
            raise RuntimeError(
                "onnxruntime is not installed. Attach an onnxruntime wheel dataset and pass --wheel_path."
            ) from None

    wheel = Path(wheel_path)
    if not wheel.is_file():
        raise FileNotFoundError(f"onnxruntime wheel not found: {wheel}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index", str(wheel)],
        check=True,
    )


def run_infer(command: list[str]) -> None:
    print("Running:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def write_placeholder_submission(sample_submission: Path, output_path: Path) -> None:
    submission = pd.read_csv(sample_submission)
    score_columns = [column for column in submission.columns if column != "row_id"]
    submission[score_columns] = 0.0
    submission.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    ensure_onnxruntime(args.wheel_path)

    artifact_dir = Path(args.artifact_dir)
    competition_dir = Path(args.competition_dir)
    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    infer_script = artifact_dir / "infer_kaggle.py"
    model_path = artifact_dir / "student.int8.onnx"
    class_list_path = artifact_dir / "classes.txt"
    test_dir = competition_dir / "test_soundscapes"
    sample_submission = competition_dir / "sample_submission.csv"

    hidden_test_files = sorted(test_dir.glob("*.ogg"))
    print(f"hidden test files: {len(hidden_test_files)}")

    if hidden_test_files:
        submission_path = working_dir / "submission.csv"
        run_infer(
            [
                sys.executable,
                str(infer_script),
                "--model_path",
                str(model_path),
                "--class_list_path",
                str(class_list_path),
                "--audio_dir",
                str(test_dir),
                "--sample_submission_csv",
                str(sample_submission),
                "--output_csv",
                str(submission_path),
            ]
        )
        print(f"Wrote {submission_path}")
        return

    train_files = sorted((competition_dir / "train_audio").rglob("*.ogg"))[: args.dryrun_n_files]
    if not train_files:
        raise RuntimeError("No train_audio files found for dry run.")

    dryrun_manifest = working_dir / "dryrun_manifest.csv"
    dryrun_predictions = working_dir / "dryrun_predictions.csv"
    submission_path = working_dir / "submission.csv"
    dryrun = pd.DataFrame(
        {
            "soundscape_id": [path.stem for path in train_files],
            "audio_path": [str(path) for path in train_files],
        }
    )
    dryrun.to_csv(dryrun_manifest, index=False)

    run_infer(
        [
            sys.executable,
            str(infer_script),
            "--model_path",
            str(model_path),
            "--class_list_path",
            str(class_list_path),
            "--test_csv",
            str(dryrun_manifest),
            "--output_csv",
            str(dryrun_predictions),
        ]
    )
    write_placeholder_submission(sample_submission, submission_path)
    print(f"Dry run complete: {dryrun_predictions}")
    print(f"Wrote placeholder {submission_path} for notebook submission compatibility.")
    print("Kaggle's hidden rerun should overwrite submission.csv with real predictions.")


if __name__ == "__main__":
    main()
