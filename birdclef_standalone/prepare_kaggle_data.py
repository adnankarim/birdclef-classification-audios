from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

from birdclef.utils import ensure_dir
from dataset import build_audio_dir_manifest, normalize_training_metadata

TRAIN_METADATA_CANDIDATES = ("train_metadata.csv", "train_labels.csv", "train.csv")
TRAIN_AUDIO_DIR_CANDIDATES = ("train_soundscapes", "train_audio")
TEST_AUDIO_DIR_CANDIDATES = ("test_soundscapes", "test_audio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize BirdCLEF Kaggle competition inputs.")
    parser.add_argument("--competition", default="birdclef-2026")
    parser.add_argument("--data_dir", required=True, help="Directory used for Kaggle downloads and extraction.")
    parser.add_argument("--output_dir", required=True, help="Directory for normalized CSVs used by the standalone pipeline.")
    parser.add_argument(
        "--competition_root",
        default=None,
        help="Existing extracted competition directory. Defaults to data_dir after any download/extraction.",
    )
    parser.add_argument("--download", action="store_true", help="Download the competition with the Kaggle CLI before preparing CSVs.")
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--keep_zips", action="store_true")
    return parser.parse_args()


def run_kaggle_download(competition: str, data_dir: Path, force_download: bool) -> None:
    command = [sys.executable, "-m", "kaggle", "competitions", "download", "-c", competition, "-p", str(data_dir)]
    if force_download:
        command.append("--force")
    subprocess.run(command, check=True)


def extract_archives(data_dir: Path, keep_zips: bool) -> None:
    for zip_path in sorted(data_dir.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(data_dir)
        if not keep_zips:
            zip_path.unlink()


def find_first_existing(root: Path, names: tuple[str, ...], *, want_dir: bool) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_dir() if want_dir else candidate.is_file():
            return candidate
    for name in names:
        matches = sorted(root.rglob(name))
        for candidate in matches:
            if candidate.is_dir() if want_dir else candidate.is_file():
                return candidate
    return None


def detect_competition_root(base_dir: Path) -> Path:
    markers = TRAIN_METADATA_CANDIDATES + TRAIN_AUDIO_DIR_CANDIDATES + TEST_AUDIO_DIR_CANDIDATES + ("sample_submission.csv",)
    if any((base_dir / marker).exists() for marker in markers):
        return base_dir
    for child in sorted(base_dir.iterdir()):
        if child.is_dir() and any((child / marker).exists() for marker in markers):
            return child
    raise FileNotFoundError(f"Could not find extracted competition files under {base_dir}.")


def main() -> None:
    args = parse_args()
    data_dir = ensure_dir(args.data_dir)
    output_dir = ensure_dir(args.output_dir)

    if args.download:
        run_kaggle_download(args.competition, data_dir, args.force_download)
        extract_archives(data_dir, keep_zips=args.keep_zips)

    competition_root = Path(args.competition_root) if args.competition_root else detect_competition_root(data_dir)

    train_csv = find_first_existing(competition_root, TRAIN_METADATA_CANDIDATES, want_dir=False)
    if train_csv is None:
        raise FileNotFoundError(
            "Could not find a training metadata CSV. Expected one of: "
            + ", ".join(TRAIN_METADATA_CANDIDATES)
        )
    train_audio_dir = find_first_existing(competition_root, TRAIN_AUDIO_DIR_CANDIDATES, want_dir=True)
    train_metadata = normalize_training_metadata(
        pd.read_csv(train_csv),
        audio_base_dir=train_audio_dir,
        require_audio=True,
        require_labels=True,
    )
    train_metadata.to_csv(output_dir / "train_metadata.csv", index=False)

    test_audio_dir = find_first_existing(competition_root, TEST_AUDIO_DIR_CANDIDATES, want_dir=True)
    if test_audio_dir is None:
        raise FileNotFoundError(
            "Could not find a test audio directory. Expected one of: " + ", ".join(TEST_AUDIO_DIR_CANDIDATES)
        )
    build_audio_dir_manifest(test_audio_dir).to_csv(output_dir / "test_metadata.csv", index=False)

    sample_submission = find_first_existing(competition_root, ("sample_submission.csv",), want_dir=False)
    if sample_submission is None:
        raise FileNotFoundError("Could not find sample_submission.csv in the competition data.")
    shutil.copy2(sample_submission, output_dir / "sample_submission.csv")

    print(f"Prepared training metadata: {output_dir / 'train_metadata.csv'}")
    print(f"Prepared test metadata: {output_dir / 'test_metadata.csv'}")
    print(f"Copied sample submission: {output_dir / 'sample_submission.csv'}")


if __name__ == "__main__":
    main()
