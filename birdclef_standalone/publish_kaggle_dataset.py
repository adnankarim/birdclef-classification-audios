from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or version a Kaggle dataset using the Kaggle CLI.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--dataset_id", required=True, help="Kaggle dataset id, e.g. username/dataset-slug")
    parser.add_argument("--title", required=True)
    parser.add_argument("--version_message", default="Update dataset")
    parser.add_argument("--license_name", default="CC0-1.0")
    parser.add_argument("--dir_mode", choices=["skip", "zip", "tar"], default="zip")
    parser.add_argument("--create", action="store_true", help="Force dataset creation instead of versioning.")
    return parser.parse_args()


def run_kaggle(command: list[str]) -> None:
    kaggle_bin = shutil.which("kaggle")
    commands: list[list[str]] = []
    if kaggle_bin:
        commands.append([kaggle_bin] + command)
    commands.append([sys.executable, "-m", "kaggle.cli"] + command)

    last_error: subprocess.CalledProcessError | None = None
    for candidate in commands:
        try:
            subprocess.run(candidate, check=True)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
    raise RuntimeError("Failed to execute Kaggle CLI command.") from last_error


def ensure_metadata(dataset_dir: Path, dataset_id: str, title: str, license_name: str) -> Path:
    metadata_path = dataset_dir / "dataset-metadata.json"
    payload = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": license_name}],
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metadata_path


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    ensure_metadata(dataset_dir, dataset_id=args.dataset_id, title=args.title, license_name=args.license_name)

    if args.create:
        run_kaggle(["datasets", "create", "-p", str(dataset_dir), "--dir-mode", args.dir_mode])
    else:
        try:
            run_kaggle(["datasets", "version", "-p", str(dataset_dir), "-m", args.version_message, "--dir-mode", args.dir_mode])
        except RuntimeError:
            run_kaggle(["datasets", "create", "-p", str(dataset_dir), "--dir-mode", args.dir_mode])


if __name__ == "__main__":
    main()
