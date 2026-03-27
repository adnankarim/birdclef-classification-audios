from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

from birdclef.utils import ensure_dir
from dataset import DEFAULT_AUDIO_EXTENSIONS, build_audio_dir_manifest, normalize_training_metadata

TRAIN_METADATA_CANDIDATES = ("train_metadata.csv", "train_labels.csv", "train.csv")
TRAIN_AUDIO_DIR_CANDIDATES = ("train_audio", "train_soundscapes")
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
    commands: list[list[str]] = []
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin:
        commands.append([kaggle_bin, "competitions", "download", "-c", competition, "-p", str(data_dir)])
    commands.append([sys.executable, "-m", "kaggle.cli", "competitions", "download", "-c", competition, "-p", str(data_dir)])

    if force_download:
        for command in commands:
            command.append("--force")

    last_error: subprocess.CalledProcessError | None = None
    for command in commands:
        try:
            subprocess.run(command, check=True)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc

    raise RuntimeError(
        "Failed to download competition data with the Kaggle CLI. "
        "Make sure the `kaggle` command works and API credentials are configured."
    ) from last_error


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


def resolve_candidate_audio_path(audio_dir: Path, raw_value: object) -> Path:
    raw_text = str(raw_value).strip()
    joined = audio_dir / raw_text
    if joined.is_file():
        return joined
    if joined.suffix:
        return joined
    for ext in DEFAULT_AUDIO_EXTENSIONS:
        candidate = joined.with_suffix(ext)
        if candidate.is_file():
            return candidate
    return joined


def pick_training_audio_dir(competition_root: Path, metadata: pd.DataFrame) -> Path | None:
    candidates = [
        path
        for name in TRAIN_AUDIO_DIR_CANDIDATES
        for path in [find_first_existing(competition_root, (name,), want_dir=True)]
        if path is not None
    ]
    if len(candidates) <= 1:
        return candidates[0] if candidates else None

    sample_series = None
    for column in ("audio_path", "filename", "path", "filepath"):
        if column in metadata.columns:
            series = metadata[column].dropna().astype(str).str.strip()
            series = series[series != ""]
            if not series.empty:
                sample_series = series.head(256)
                break
    if sample_series is None:
        return candidates[0]

    best_dir = candidates[0]
    best_score = -1
    for candidate_dir in candidates:
        score = sum(resolve_candidate_audio_path(candidate_dir, value).is_file() for value in sample_series)
        if score > best_score:
            best_score = score
            best_dir = candidate_dir
    return best_dir


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
    train_metadata_raw = pd.read_csv(train_csv)
    train_audio_dir = pick_training_audio_dir(competition_root, train_metadata_raw)
    train_metadata = normalize_training_metadata(
        train_metadata_raw,
        audio_base_dir=train_audio_dir,
        require_audio=True,
        require_labels=True,
    )
    train_metadata.to_csv(output_dir / "train_metadata.csv", index=False)
    print(f"Using training audio directory: {train_audio_dir}")

    test_audio_dir = find_first_existing(competition_root, TEST_AUDIO_DIR_CANDIDATES, want_dir=True)
    if test_audio_dir is None:
        raise FileNotFoundError(
            "Could not find a test audio directory. Expected one of: " + ", ".join(TEST_AUDIO_DIR_CANDIDATES)
        )
    test_manifest = build_audio_dir_manifest(test_audio_dir)
    if test_manifest.empty:
        raise RuntimeError(
            f"Found test audio directory at {test_audio_dir}, but no audio files were discovered under it. "
            "This usually means the downloaded competition files do not include the test soundscapes for local inference. "
            "Run inference in the Kaggle competition notebook environment or point infer_kaggle.py at the actual test audio directory."
        )
    test_manifest.to_csv(output_dir / "test_metadata.csv", index=False)

    sample_submission = find_first_existing(competition_root, ("sample_submission.csv",), want_dir=False)
    if sample_submission is None:
        raise FileNotFoundError("Could not find sample_submission.csv in the competition data.")
    shutil.copy2(sample_submission, output_dir / "sample_submission.csv")

    print(f"Prepared training metadata: {output_dir / 'train_metadata.csv'}")
    print(f"Prepared test metadata: {output_dir / 'test_metadata.csv'}")
    print(f"Copied sample submission: {output_dir / 'sample_submission.csv'}")


if __name__ == "__main__":
    main()
