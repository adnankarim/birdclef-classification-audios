from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from birdclef.utils import ensure_dir
from dataset import load_manifest, resolve_class_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend multiple pseudo-label CSVs into one training-ready CSV.")
    parser.add_argument("--reference_window_manifest_csv", required=True)
    parser.add_argument("--class_list_path", required=True)
    parser.add_argument("--input_csv", nargs="+", required=True)
    parser.add_argument("--weights", nargs="*", type=float, default=None)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--keep_threshold", type=float, default=0.7)
    parser.add_argument("--label_threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classes = resolve_class_list(pd.DataFrame({"labels": []}), args.class_list_path)
    reference = load_manifest(args.reference_window_manifest_csv)
    key_columns = ["soundscape_id", "window_idx"]
    if not set(key_columns).issubset(reference.columns):
        raise ValueError(f"Reference manifest must contain {key_columns}.")

    if args.weights is None or len(args.weights) == 0:
        weights = np.ones(len(args.input_csv), dtype=np.float32)
    else:
        if len(args.weights) != len(args.input_csv):
            raise ValueError("--weights must match the number of --input_csv files.")
        weights = np.asarray(args.weights, dtype=np.float32)
    if (weights < 0).any():
        raise ValueError("--weights must be non-negative.")
    if float(weights.sum()) <= 0:
        raise ValueError("--weights must sum to a positive value.")

    accum: pd.DataFrame | None = None
    weight_sum: pd.Series | None = None
    input_summaries: list[dict[str, object]] = []
    for path_str, weight in zip(args.input_csv, weights.tolist()):
        frame = load_manifest(path_str)
        missing_classes = [column for column in classes if column not in frame.columns]
        if missing_classes:
            raise ValueError(f"{path_str} is missing class columns: {missing_classes[:5]}")
        current = frame[key_columns + classes].copy()
        current[classes] = current[classes].to_numpy(dtype=np.float32, copy=True)
        current = current.groupby(key_columns, as_index=False)[classes].mean().set_index(key_columns)

        if accum is None:
            accum = current * weight
            weight_sum = pd.Series(weight, index=current.index, dtype=np.float32)
        else:
            accum = accum.add(current * weight, fill_value=0.0)
            weight_sum = weight_sum.add(pd.Series(weight, index=current.index, dtype=np.float32), fill_value=0.0)

        input_summaries.append({"path": str(Path(path_str).resolve()), "rows": int(len(current)), "weight": float(weight)})

    assert accum is not None and weight_sum is not None
    averaged = accum.div(weight_sum.clip(lower=1e-8), axis=0).reset_index()
    blended = averaged.merge(reference, on=key_columns, how="left")
    missing_reference = blended["spec_path"].isna().sum() if "spec_path" in blended.columns else 0
    if missing_reference:
        raise ValueError(f"{missing_reference} blended rows could not be matched back to the reference window manifest.")

    for class_name in classes:
        blended[class_name] = blended[class_name].astype(np.float32)
    blended["confidence"] = blended[classes].max(axis=1).astype(np.float32)
    blended["labels"] = [
        " ".join([classes[idx] for idx, prob in enumerate(row) if prob >= args.label_threshold])
        for row in blended[classes].to_numpy(dtype=np.float32, copy=True)
    ]
    blended = blended[blended["confidence"] >= args.keep_threshold].reset_index(drop=True)

    output_csv = Path(args.output_csv)
    ensure_dir(output_csv.parent)
    blended.to_csv(output_csv, index=False)

    summary = {
        "output_csv": str(output_csv.resolve()),
        "num_inputs": len(args.input_csv),
        "num_blended_rows": int(len(blended)),
        "keep_threshold": args.keep_threshold,
        "label_threshold": args.label_threshold,
        "inputs": input_summaries,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output_json:
        output_json = Path(args.output_json)
        ensure_dir(output_json.parent)
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote report: {output_json}")


if __name__ == "__main__":
    main()
