from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from birdclef.audio import AudioParams, compute_logmel, count_windows, load_audio_mono
from birdclef.models import PerchEmbeddingExtractor
from birdclef.utils import ensure_dir
from dataset import (
    build_window_manifest,
    normalize_training_metadata,
    resolve_class_list,
    save_class_list,
    save_window_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute BirdCLEF-style log-mel soundscape features.")
    parser.add_argument("--metadata_csv", required=True, help="CSV with at least soundscape_id/audio_path/labels columns.")
    parser.add_argument("--output_dir", required=True, help="Directory for spectrograms, manifests, and optional Perch embeddings.")
    parser.add_argument("--audio_column", default="audio_path")
    parser.add_argument("--id_column", default="soundscape_id")
    parser.add_argument("--label_column", default="labels")
    parser.add_argument("--audio_base_dir", default=None, help="Base directory used to resolve relative audio filenames.")
    parser.add_argument("--strong_start_column", default=None)
    parser.add_argument("--strong_end_column", default=None)
    parser.add_argument("--compute_perch_embeddings", action="store_true")
    parser.add_argument("--fail_on_bad_audio", action="store_true", help="Stop immediately when an audio file cannot be decoded.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = AudioParams()
    output_dir = ensure_dir(args.output_dir)
    spec_dir = ensure_dir(output_dir / "spectrograms")
    perch_dir = ensure_dir(output_dir / "perch_embeddings")

    metadata = normalize_training_metadata(
        pd.read_csv(args.metadata_csv),
        audio_column=args.audio_column,
        id_column=args.id_column,
        label_column=args.label_column,
        audio_base_dir=args.audio_base_dir,
        require_audio=True,
        require_labels=True,
    )

    class_list = resolve_class_list(metadata, label_column=args.label_column)
    save_class_list(class_list, output_dir / "classes.txt")

    perch_extractor = PerchEmbeddingExtractor() if args.compute_perch_embeddings else None

    file_metadata = metadata[["soundscape_id", "audio_path"]].drop_duplicates("soundscape_id").reset_index(drop=True)
    file_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, str]] = []
    for row in tqdm(file_metadata.itertuples(index=False), total=len(file_metadata), desc="preprocess"):
        audio_path = Path(row.audio_path)
        soundscape_id = getattr(row, "soundscape_id")
        try:
            waveform = load_audio_mono(audio_path, sample_rate=params.sample_rate)
        except Exception as exc:
            if args.fail_on_bad_audio:
                raise
            skipped_rows.append(
                {
                    "soundscape_id": str(soundscape_id),
                    "audio_path": str(audio_path),
                    "error": str(exc),
                }
            )
            continue
        logmel = compute_logmel(waveform, params).cpu().numpy().astype(np.float16)
        spec_path = spec_dir / f"{soundscape_id}.npy"
        np.save(spec_path, logmel)

        perch_path = ""
        if perch_extractor is not None:
            window_waveforms: list[np.ndarray] = []
            window_size = params.window_num_samples
            stride = params.stride_num_samples
            max_start = max(len(waveform) - window_size, 0)
            starts = list(range(0, max_start + 1, max(stride, 1)))
            if not starts or starts[-1] != max_start:
                starts.append(max_start)
            for start in starts:
                segment = waveform[start : start + window_size]
                if len(segment) < window_size:
                    segment = np.pad(segment, (0, window_size - len(segment)))
                window_waveforms.append(segment.astype(np.float32))
            embeddings = perch_extractor.embed_windows(window_waveforms, sample_rate=params.sample_rate)
            perch_path = str(perch_dir / f"{soundscape_id}.npy")
            np.save(perch_path, embeddings.astype(np.float32))

        file_rows.append(
            {
                "soundscape_id": soundscape_id,
                "audio_path": str(audio_path),
                "spec_path": str(spec_path),
                "perch_embedding_path": perch_path,
                "num_frames": int(logmel.shape[1]),
                "num_windows": count_windows(int(logmel.shape[1]), params),
                "duration_sec": float(len(waveform) / params.sample_rate),
            }
        )

    file_manifest = pd.DataFrame(file_rows)
    file_manifest.to_csv(output_dir / "manifest.csv", index=False)
    if skipped_rows:
        skipped_manifest = pd.DataFrame(skipped_rows)
        skipped_manifest.to_csv(output_dir / "skipped_audio.csv", index=False)
        print(f"Skipped {len(skipped_manifest)} unreadable audio files. Details: {output_dir / 'skipped_audio.csv'}")
    if file_manifest.empty:
        raise RuntimeError("No audio files were successfully preprocessed.")

    window_manifest = build_window_manifest(
        metadata=metadata,
        file_manifest=file_manifest,
        params=params,
        label_column=args.label_column,
        strong_start_column=args.strong_start_column,
        strong_end_column=args.strong_end_column,
    )
    save_window_manifest(window_manifest, output_dir / "window_manifest.csv")
    print(f"Prepared {len(file_manifest)} audio files and {len(window_manifest)} windows.")


if __name__ == "__main__":
    main()
