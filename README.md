# BirdCLEF+ 2026 Soundscape Pipeline

This repository contains a standalone BirdCLEF+ 2026 multilabel audio classification pipeline. The main working code lives in [`birdclef_standalone/`](birdclef_standalone/).

The project has three practical goals:

- Train and distill BirdCLEF soundscape models from log-mel and Perch embeddings.
- Export fast Kaggle-compatible ONNX/INT8 student models.
- Track local validation and Kaggle submission results so model changes are measurable.

## Current Status

The most reliable Kaggle submission path so far is the ONNX student pipeline. A reduced teacher submission also runs successfully, but full teacher inference timed out under Kaggle CPU limits.

The Perch temporal student trains and validates well on ROC-AUC, but the current CNN student still wins on local fold-0 macro AP. The best next modeling direction is to use Perch as an additional distillation signal rather than replacing the CNN student directly.

## Results So Far

### Kaggle Public Scores

| Submission | Path | Status | Public Score | Notes |
|---|---:|---:|---:|---|
| `BirdCLEF+ 2026 Inference - Version 1` | reference notebook | Succeeded | `0.907` | External/reference Perch-style notebook baseline. |
| `notebookb21eefcf68 - Version 2` | student INT8 ONNX | Succeeded | `0.720` | First working offline Kaggle submission. |
| `notebookb21eefcf68 - Version 3` | full teacher ensemble | Timeout | n/a | Too slow for Kaggle 90 minute CPU limit. |
| `notebookb21eefcf68 - Version 5` | top-3 PyTorch teachers | Succeeded | `0.736` | Faster teacher subset; modest improvement over INT8 student. |

### Local Fold-0 Validation

Both rows below were evaluated with `evaluate_model.py` on validation fold 0.

| Model | Checkpoint | Macro AP | Macro ROC-AUC | Micro ROC-AUC | Loss | Windows |
|---|---|---:|---:|---:|---:|---:|
| CNN student | `outputs/birdclef_student_round2/student_best.pth` | `0.961097` | `0.994790` | `0.996315` | `0.001541` | `43,632` |
| Perch temporal student | `outputs/birdclef_perch_student/perch_student_best.pth` | `0.947069` | `0.997049` | `0.998110` | `0.002144` | `43,632` |

Interpretation: Perch has stronger ROC-AUC, but the CNN student has better macro AP. For the current competition scoring behavior, macro AP is the safer local metric to prioritize.

### ONNX FP32 vs INT8 Quantization

Measured with `compare_onnx_models.py` on a 128-file labeled sample.

| Model | Macro AP | Macro ROC-AUC | Micro ROC-AUC |
|---|---:|---:|---:|
| FP32 ONNX student | `0.974490` | `0.976172` | `0.983115` |
| INT8 ONNX student | `0.960258` | `0.976016` | `0.982060` |
| INT8 - FP32 delta | `-0.014232` | `-0.000156` | `-0.001055` |

Interpretation: quantization caused a small ROC-AUC drop and a more noticeable macro AP drop. If runtime allows, FP32 ONNX is worth testing as a Kaggle submission.

### Teacher vs Student Comparison

Measured with `compare_teacher_student_models.py` on the same 128-file sample.

| Model | Macro AP | Macro ROC-AUC | Micro ROC-AUC |
|---|---:|---:|---:|
| Teacher ensemble | `0.997222` | `0.999912` | `0.999901` |
| FP32 student | `0.974490` | `0.976172` | `0.983115` |
| INT8 student | `0.960258` | `0.976016` | `0.982060` |

Important caveat: the teacher numbers are optimistic because the comparison is not strict out-of-fold for every sampled file. The direction is still useful: most loss is from teacher-to-student distillation, while INT8 adds a smaller additional loss.

### Perch Preprocessing and Training

| Item | Result |
|---|---:|
| Perch preprocessing completed | `35,549` audio files |
| Prepared windows | `266,049` |
| Runtime observed | about `8h 25m` |
| Perch embedding dim | `1536` |
| Perch temporal checkpoint | `outputs/birdclef_perch_student/perch_student_best.pth` |
| Best epoch | `15` |
| Train loss | `0.127432` |
| Validation loss | `0.002144` |
| Max windows in sequence model | `1377` |

## Repository Layout

```text
birdclef/
  README.md
  birdclef-2026-inference.ipynb
  birdclef_standalone/
    birdclef/                         # audio, models, training utilities
    prepare_kaggle_data.py
    preprocess.py
    train_teachers.py
    train_student.py
    train_perch_student.py
    generate_pseudolabels.py
    generate_perch_sequence_pseudolabels.py
    blend_pseudolabels.py
    evaluate_model.py
    export_int8.py
    infer_kaggle.py
    kaggle_submission_runner.py
    teacher_kaggle_submission_runner.py
```

## Setup

Install the Python dependencies:

```bash
cd birdclef_standalone
python3 -m pip install -r requirements.txt
```

For Perch embeddings, install the extra packages:

```bash
python3 -m pip install --user --upgrade "bioacoustics-model-zoo[tensorflow]" tensorflow-hub
python3 -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
python3 -m pip install --user --upgrade opencv-python-headless
```

The Perch package may download model weights the first time it is used, so run that step in an environment with internet enabled.

## Main Workflow

All commands below assume:

```bash
cd ~/birdclef/birdclef_standalone
```

### 1. Prepare Data

```bash
python3 prepare_kaggle_data.py \
  --download \
  --competition birdclef-2026 \
  --data_dir data/raw \
  --output_dir data
```

### 2. Preprocess Log-Mel Features

```bash
python3 preprocess.py \
  --metadata_csv data/train_metadata.csv \
  --output_dir birdclef_prepared
```

For Perch training, include cached Perch embeddings:

```bash
python3 preprocess.py \
  --metadata_csv data/train_metadata.csv \
  --output_dir birdclef_prepared \
  --compute_perch_embeddings
```

### 3. Train Teacher Ensemble

```bash
python3 train_teachers.py \
  --metadata_csv data/train_metadata.csv \
  --window_manifest_csv birdclef_prepared/window_manifest.csv \
  --class_list_path birdclef_prepared/classes.txt \
  --output_dir outputs/birdclef_teachers \
  --use_mil
```

Optional stronger heterogeneous teachers:

```bash
python3 train_teachers.py \
  --metadata_csv data/train_metadata.csv \
  --window_manifest_csv birdclef_prepared/window_manifest.csv \
  --class_list_path birdclef_prepared/classes.txt \
  --output_dir outputs/birdclef_teachers_sota \
  --teacher_model_types efficientnet_v2_m convnext_small \
  --train_perch_teacher \
  --use_mil
```

### 4. Generate Teacher Pseudo Labels

```bash
python3 generate_pseudolabels.py \
  --teacher_manifest_json outputs/birdclef_teachers/teacher_manifest.json \
  --window_manifest_csv birdclef_prepared/window_manifest.csv \
  --output_dir outputs/pseudo_round1 \
  --round_idx 1
```

### 5. Train Perch Temporal Student

```bash
python3 train_perch_student.py \
  --metadata_csv data/train_metadata.csv \
  --labeled_window_manifest_csv birdclef_prepared/window_manifest.csv \
  --pseudo_label_csv outputs/pseudo_round1/pseudo_labels_round1.csv \
  --class_list_path birdclef_prepared/classes.txt \
  --output_dir outputs/birdclef_perch_student \
  --hidden_dim 768 \
  --num_heads 8 \
  --num_layers 4 \
  --epochs 16
```

### 6. Generate Perch Pseudo Labels and Blend

```bash
python3 generate_perch_sequence_pseudolabels.py \
  --checkpoint_path outputs/birdclef_perch_student/perch_student_best.pth \
  --window_manifest_csv birdclef_prepared/window_manifest.csv \
  --class_list_path birdclef_prepared/classes.txt \
  --output_dir outputs/pseudo_perch_round1 \
  --round_idx 1
```

```bash
python3 blend_pseudolabels.py \
  --reference_window_manifest_csv birdclef_prepared/window_manifest.csv \
  --class_list_path birdclef_prepared/classes.txt \
  --input_csv outputs/pseudo_round1/pseudo_labels_round1.csv outputs/pseudo_perch_round1/pseudo_labels_round1.csv \
  --weights 0.7 0.3 \
  --output_csv outputs/pseudo_blend_round1/pseudo_labels_round1_blended.csv \
  --output_json outputs/pseudo_blend_round1/pseudo_labels_round1_blended.json
```

### 7. Train CNN Student

Baseline pseudo labels:

```bash
python3 train_student.py \
  --metadata_csv data/train_metadata.csv \
  --labeled_window_manifest_csv birdclef_prepared/window_manifest.csv \
  --pseudo_label_csv outputs/pseudo_round1/pseudo_labels_round1.csv \
  --class_list_path birdclef_prepared/classes.txt \
  --output_dir outputs/birdclef_student_round1 \
  --use_mil
```

Blended teacher plus Perch pseudo labels:

```bash
python3 train_student.py \
  --metadata_csv data/train_metadata.csv \
  --labeled_window_manifest_csv birdclef_prepared/window_manifest.csv \
  --pseudo_label_csv outputs/pseudo_blend_round1/pseudo_labels_round1_blended.csv \
  --class_list_path birdclef_prepared/classes.txt \
  --output_dir outputs/birdclef_student_blend_round1 \
  --use_mil
```

### 8. Evaluate and Log Models

```bash
python3 evaluate_model.py \
  --checkpoint_path outputs/birdclef_student_round2/student_best.pth \
  --metadata_csv data/train_metadata.csv \
  --window_manifest_csv birdclef_prepared/window_manifest.csv \
  --class_list_path birdclef_prepared/classes.txt \
  --validation_fold 0 \
  --run_label cnn_student_fold0 \
  --output_json outputs/birdclef_student_round2/eval_fold0.json
```

```bash
python3 evaluate_model.py \
  --checkpoint_path outputs/birdclef_perch_student/perch_student_best.pth \
  --metadata_csv data/train_metadata.csv \
  --window_manifest_csv birdclef_prepared/window_manifest.csv \
  --class_list_path birdclef_prepared/classes.txt \
  --validation_fold 0 \
  --run_label perch_temporal_fold0 \
  --output_json outputs/birdclef_perch_student/eval_fold0.json
```

The evaluator appends local results to:

```text
outputs/eval_runs/model_eval_log.csv
outputs/eval_runs/model_eval_log.jsonl
```

### 9. Export Student to ONNX and INT8

```bash
python3 export_int8.py \
  --student_checkpoint outputs/birdclef_student_round1/student_best.pth \
  --calibration_window_manifest_csv birdclef_prepared/window_manifest.csv \
  --output_dir outputs/birdclef_export
```

### 10. Package Kaggle Student Artifacts

```bash
python3 package_kaggle_artifacts.py \
  --model_path outputs/birdclef_export/student.int8.onnx \
  --class_list_path birdclef_prepared/classes.txt \
  --output_dir final_artifacts \
  --zip
```

Upload `final_artifacts.zip` to Kaggle as a dataset.

### 11. Kaggle Notebook Student Submission

Attach these datasets:

- BirdCLEF+ 2026 competition dataset
- student artifact dataset
- `onnxruntime` wheel dataset for offline install

Then run:

```python
!python3 /kaggle/input/datasets/adnankarem/dataclef/kaggle_submission_runner.py \
  --artifact_dir /kaggle/input/datasets/adnankarem/dataclef \
  --competition_dir /kaggle/input/competitions/birdclef-2026 \
  --wheel_path /kaggle/input/datasets/adnankarem/onnxruntime-wheel/onnxruntime-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

Disable internet, save a version, and submit the notebook. Kaggle injects hidden test audio during the private rerun.

## Teacher Submission Notes

Full teacher inference timed out. The reduced teacher path works with top-N checkpoints:

```python
!python3 /kaggle/input/datasets/adnankarem/birdclef-teacher-artifacts/teacher_kaggle_submission_runner.py \
  --artifact_dir /kaggle/input/datasets/adnankarem/birdclef-teacher-artifacts \
  --competition_dir /kaggle/input/competitions/birdclef-2026 \
  --max_teachers 3 \
  --device cpu
```

This produced a public score of `0.736`, better than the first INT8 student but still below the reference Perch notebook.

## Important Lessons

- The Kaggle interactive notebook only shows `test_soundscapes/readme.txt`; hidden test audio appears only during notebook submission reruns.
- Offline Kaggle submissions cannot use internet. Dependencies such as `onnxruntime` must be uploaded as Kaggle datasets or already present.
- Runtime teacher INT8 conversion is not recommended. It burns Kaggle CPU time and makes timeout risk worse.
- The current Perch temporal student should be used as a distillation teacher, not as the direct main submission model.
- Macro AP has been the most useful local discriminator so far.

## Current Next Step

Finish and evaluate the blended CNN student:

```bash
python3 evaluate_model.py \
  --checkpoint_path outputs/birdclef_student_blend_round1/student_best.pth \
  --metadata_csv data/train_metadata.csv \
  --window_manifest_csv birdclef_prepared/window_manifest.csv \
  --class_list_path birdclef_prepared/classes.txt \
  --validation_fold 0 \
  --run_label cnn_student_blended_fold0 \
  --output_json outputs/birdclef_student_blend_round1/eval_fold0.json
```

If it improves macro AP over `0.961097`, export and submit it. If it does not, stay with the current CNN student pipeline and test FP32 ONNX on Kaggle if runtime allows.
