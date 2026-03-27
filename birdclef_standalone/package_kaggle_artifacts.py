from __future__ import annotations

import argparse
import shutil
from pathlib import Path


README_TEXT = """BirdCLEF Kaggle Inference Bundle
================================

This bundle contains the exported model and helper files needed to run
final inference in a Kaggle notebook.

Files:
- student.int8.onnx
- classes.txt
- infer_kaggle.py
- kaggle_submission_runner.py
- dataset.py
- birdclef/

Kaggle notebook cells:

Cell 1
------
!ls /kaggle/input
!find /kaggle/input -maxdepth 2 -type f | head -50

Cell 2
------
ARTIFACT_DIR = "/kaggle/input/YOUR_DATASET_NAME"
COMP_DIR = "/kaggle/input/competitions/birdclef-2026"

Cell 3
------
!python3 {ARTIFACT_DIR}/kaggle_submission_runner.py \
  --artifact_dir {ARTIFACT_DIR} \
  --competition_dir {COMP_DIR} \
  --wheel_path /kaggle/input/YOUR_WHEEL_DATASET/onnxruntime-*.whl

Cell 4
------
from pathlib import Path
print("submission:", Path("/kaggle/working/submission.csv").exists())
print("dryrun:", Path("/kaggle/working/dryrun_predictions.csv").exists())

Replace YOUR_DATASET_NAME with the actual Kaggle dataset folder name shown in
Cell 1.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package BirdCLEF inference artifacts for upload to Kaggle."
    )
    parser.add_argument(
        "--model_path",
        default="outputs/birdclef_export/student.int8.onnx",
        help="Path to the exported ONNX model.",
    )
    parser.add_argument(
        "--class_list_path",
        default="birdclef_prepared/classes.txt",
        help="Path to classes.txt used by inference.",
    )
    parser.add_argument(
        "--infer_script_path",
        default="infer_kaggle.py",
        help="Path to infer_kaggle.py.",
    )
    parser.add_argument(
        "--runner_script_path",
        default="kaggle_submission_runner.py",
        help="Path to kaggle_submission_runner.py.",
    )
    parser.add_argument(
        "--dataset_module_path",
        default="dataset.py",
        help="Path to dataset.py used by infer_kaggle.py.",
    )
    parser.add_argument(
        "--package_dir",
        default="birdclef",
        help="Path to the local birdclef package directory.",
    )
    parser.add_argument(
        "--output_dir",
        default="final_artifacts",
        help="Directory to write the Kaggle upload bundle.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also create a zip archive next to output_dir.",
    )
    return parser.parse_args()


def require_file(path_str: str, label: str) -> Path:
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def require_dir(path_str: str, label: str) -> Path:
    path = Path(path_str)
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def main() -> None:
    args = parse_args()
    model_path = require_file(args.model_path, "Model")
    class_list_path = require_file(args.class_list_path, "Class list")
    infer_script_path = require_file(args.infer_script_path, "Inference script")
    runner_script_path = require_file(args.runner_script_path, "Runner script")
    dataset_module_path = require_file(args.dataset_module_path, "Dataset module")
    package_dir = require_dir(args.package_dir, "birdclef package")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(model_path, output_dir / "student.int8.onnx")
    shutil.copy2(class_list_path, output_dir / "classes.txt")
    shutil.copy2(infer_script_path, output_dir / "infer_kaggle.py")
    shutil.copy2(runner_script_path, output_dir / "kaggle_submission_runner.py")
    shutil.copy2(dataset_module_path, output_dir / "dataset.py")
    packaged_module_dir = output_dir / "birdclef"
    if packaged_module_dir.exists():
        shutil.rmtree(packaged_module_dir)
    shutil.copytree(package_dir, packaged_module_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (output_dir / "README.txt").write_text(README_TEXT, encoding="utf-8")

    print(f"Wrote Kaggle artifacts to: {output_dir}")
    for filename in [
        "student.int8.onnx",
        "classes.txt",
        "infer_kaggle.py",
        "kaggle_submission_runner.py",
        "dataset.py",
        "README.txt",
    ]:
        file_path = output_dir / filename
        print(f"- {file_path} ({file_path.stat().st_size} bytes)")
    print(f"- {packaged_module_dir} (package directory)")

    if args.zip:
        archive_path = shutil.make_archive(str(output_dir), "zip", root_dir=output_dir)
        print(f"Created zip archive: {archive_path}")


if __name__ == "__main__":
    main()
