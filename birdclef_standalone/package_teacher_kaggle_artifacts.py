from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


README_TEXT = """BirdCLEF Teacher Kaggle Inference Bundle
========================================

This bundle contains the teacher ensemble checkpoints and helper files needed
to run teacher inference in a Kaggle notebook.

Files:
- teacher_manifest.json
- classes.txt
- infer_teacher_kaggle.py
- teacher_kaggle_submission_runner.py
- dataset.py
- birdclef/
- fold_*/teacher checkpoints

Kaggle notebook cells:

Cell 1
------
!python3 /kaggle/input/YOUR_TEACHER_DATASET/teacher_kaggle_submission_runner.py \
  --artifact_dir /kaggle/input/YOUR_TEACHER_DATASET \
  --competition_dir /kaggle/input/competitions/birdclef-2026 \
  --device cpu

Cell 2
------
from pathlib import Path
print("submission:", Path("/kaggle/working/submission.csv").exists())
print("dryrun:", Path("/kaggle/working/dryrun_predictions.csv").exists())
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package BirdCLEF teacher artifacts for upload to Kaggle."
    )
    parser.add_argument("--teacher_manifest_json", required=True)
    parser.add_argument("--class_list_path", required=True)
    parser.add_argument("--infer_script_path", default="infer_teacher_kaggle.py")
    parser.add_argument("--runner_script_path", default="teacher_kaggle_submission_runner.py")
    parser.add_argument("--dataset_module_path", default="dataset.py")
    parser.add_argument("--package_dir", default="birdclef")
    parser.add_argument("--output_dir", default="teacher_kaggle_artifacts")
    parser.add_argument("--zip", action="store_true")
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
    teacher_manifest_path = require_file(args.teacher_manifest_json, "Teacher manifest")
    class_list_path = require_file(args.class_list_path, "Class list")
    infer_script_path = require_file(args.infer_script_path, "Teacher inference script")
    runner_script_path = require_file(args.runner_script_path, "Teacher runner script")
    dataset_module_path = require_file(args.dataset_module_path, "Dataset module")
    package_dir = require_dir(args.package_dir, "birdclef package")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
    source_manifest_dir = teacher_manifest_path.parent.resolve()

    rewritten_manifest = dict(manifest)
    rewritten_folds = []
    for fold in manifest["folds"]:
        new_fold = dict(fold)
        new_teachers = []
        for teacher in fold["teachers"]:
            new_teacher = dict(teacher)
            checkpoint_path = Path(teacher["checkpoint_path"])
            if not checkpoint_path.is_absolute():
                checkpoint_path = source_manifest_dir / checkpoint_path
            relative_checkpoint = Path(f"fold_{fold['fold']}") / checkpoint_path.name
            target_checkpoint = output_dir / relative_checkpoint
            target_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint_path, target_checkpoint)
            new_teacher["checkpoint_path"] = str(relative_checkpoint)
            new_teachers.append(new_teacher)
        new_fold["teachers"] = new_teachers
        rewritten_folds.append(new_fold)
    rewritten_manifest["folds"] = rewritten_folds

    (output_dir / "teacher_manifest.json").write_text(
        json.dumps(rewritten_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    shutil.copy2(class_list_path, output_dir / "classes.txt")
    shutil.copy2(infer_script_path, output_dir / "infer_teacher_kaggle.py")
    shutil.copy2(runner_script_path, output_dir / "teacher_kaggle_submission_runner.py")
    shutil.copy2(dataset_module_path, output_dir / "dataset.py")
    packaged_module_dir = output_dir / "birdclef"
    if packaged_module_dir.exists():
        shutil.rmtree(packaged_module_dir)
    shutil.copytree(package_dir, packaged_module_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (output_dir / "README.txt").write_text(README_TEXT, encoding="utf-8")

    print(f"Wrote teacher Kaggle artifacts to: {output_dir}")
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            print(f"- {path.relative_to(output_dir)} ({path.stat().st_size} bytes)")

    if args.zip:
        archive_path = shutil.make_archive(str(output_dir), "zip", root_dir=output_dir)
        print(f"Created zip archive: {archive_path}")


if __name__ == "__main__":
    main()
