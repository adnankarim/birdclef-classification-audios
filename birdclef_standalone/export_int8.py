from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from birdclef.audio import AudioParams
from birdclef.models import StudentExportWrapper, build_image_classifier
from birdclef.training import load_checkpoint
from birdclef.utils import ensure_dir
from dataset import BirdCLEFWindowDataset, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BirdCLEF student to ONNX and INT8.")
    parser.add_argument("--student_checkpoint", required=True)
    parser.add_argument("--calibration_window_manifest_csv", default=None)
    parser.add_argument("--class_list_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--calibration_samples", type=int, default=256)
    return parser.parse_args()


def build_model(checkpoint: dict) -> StudentExportWrapper:
    model = build_image_classifier(
        checkpoint["model_type"],
        num_classes=len(checkpoint["classes"]),
        pretrained=False,
        use_mil=bool(checkpoint.get("use_mil", False)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return StudentExportWrapper(model)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    checkpoint = load_checkpoint(args.student_checkpoint, map_location="cpu")
    model = build_model(checkpoint)
    params = AudioParams(**checkpoint["audio_params"])

    onnx_path = output_dir / "student.onnx"
    dummy = torch.randn(1, 1, params.n_mels, params.window_num_frames, dtype=torch.float32)
    try:
        import onnxscript  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch ONNX export requires `onnxscript` in this environment. "
            "Install it with `python3 -m pip install onnxscript` or reinstall from requirements.txt."
        ) from exc
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["inputs"],
        output_names=["probabilities"],
        dynamic_axes={"inputs": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported ONNX model: {onnx_path}")

    if not args.calibration_window_manifest_csv:
        return

    try:
        from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "INT8 quantization requires `onnxruntime`. "
            "Install it with `python3 -m pip install onnxruntime` or reinstall from requirements.txt. "
            f"The non-quantized ONNX model was already written to {onnx_path}."
        ) from exc

    calibration_df = load_manifest(args.calibration_window_manifest_csv).head(args.calibration_samples)
    dataset = BirdCLEFWindowDataset(calibration_df, checkpoint["classes"], params=params, training=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    class _Reader(CalibrationDataReader):
        def __init__(self, dataloader: DataLoader):
            self._iterator = iter(dataloader)

        def get_next(self):
            try:
                batch = next(self._iterator)
            except StopIteration:
                return None
            return {"inputs": batch["inputs"].numpy().astype(np.float32)}

    quantize_static(
        model_input=str(onnx_path),
        model_output=str(output_dir / "student.int8.onnx"),
        calibration_data_reader=_Reader(loader),
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        quant_format=QuantFormat.QDQ,
    )
    print(f"Exported INT8 ONNX model: {output_dir / 'student.int8.onnx'}")


if __name__ == "__main__":
    main()
