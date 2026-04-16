# BirdCLEF Standalone

Standalone BirdCLEF-style multilabel soundscape pipeline extracted from this repo.

For the full project overview, current results, and recommended next steps, see the root `README.md`.

Results so far:

| Area | Result | Notes |
|---|---:|---|
| Kaggle student INT8 ONNX | `0.720` public | `notebookb21eefcf68 - Version 2`, first working submission. |
| Kaggle top-3 teacher subset | `0.736` public | `notebookb21eefcf68 - Version 5`, full teacher ensemble timed out. |
| Reference Perch notebook | `0.907` public | External/reference notebook result. |
| CNN student fold-0 macro AP | `0.961097` | `outputs/birdclef_student_round2/student_best.pth`. |
| Perch temporal fold-0 macro AP | `0.947069` | Better ROC-AUC, lower macro AP than CNN student. |
| FP32 to INT8 macro AP delta | `-0.014232` | INT8 quantization hurt macro AP more than ROC-AUC. |

Current recommendation:
- Keep the CNN student as the main submission path.
- Use Perch as an additional distillation signal via `generate_perch_sequence_pseudolabels.py` and `blend_pseudolabels.py`.
- Evaluate the blended CNN student before packaging or submitting it.

Files:
- `prepare_kaggle_data.py`
- `preprocess.py`
- `dataset.py`
- `train_teachers.py`
- `generate_pseudolabels.py`
- `generate_perch_sequence_pseudolabels.py`
- `blend_pseudolabels.py`
- `train_student.py`
- `train_perch_student.py`
- `export_int8.py`
- `infer_kaggle.py`
- `infer_teacher_kaggle.py`
- `compare_onnx_models.py`
- `compare_teacher_student_models.py`
- `kaggle_submission_runner.py`
- `teacher_kaggle_submission_runner.py`
- `package_teacher_kaggle_artifacts.py`
- `publish_kaggle_dataset.py`
- `birdclef/`

Expected workflow:

```bash
python3 prepare_kaggle_data.py \
  --download \
  --competition birdclef-2026 \
  --data_dir data/raw \
  --output_dir data

python3 preprocess.py \
  --metadata_csv data/train_metadata.csv \
  --output_dir birdclef_prepared

python3 train_teachers.py \
  --metadata_csv data/train_metadata.csv \
  --window_manifest_csv birdclef_prepared/window_manifest.csv \
  --class_list_path birdclef_prepared/classes.txt \
  --output_dir outputs/birdclef_teachers \
  --use_mil

python3 generate_pseudolabels.py \
  --teacher_manifest_json outputs/birdclef_teachers/teacher_manifest.json \
  --window_manifest_csv birdclef_prepared/window_manifest.csv \
  --output_dir outputs/pseudo_round1 \
  --round_idx 1

python3 generate_perch_sequence_pseudolabels.py \
  --checkpoint_path outputs/birdclef_perch_student/perch_student_best.pth \
  --window_manifest_csv birdclef_prepared/window_manifest.csv \
  --class_list_path birdclef_prepared/classes.txt \
  --output_dir outputs/pseudo_perch_round1 \
  --round_idx 1

python3 blend_pseudolabels.py \
  --reference_window_manifest_csv birdclef_prepared/window_manifest.csv \
  --class_list_path birdclef_prepared/classes.txt \
  --input_csv outputs/pseudo_round1/pseudo_labels_round1.csv outputs/pseudo_perch_round1/pseudo_labels_round1.csv \
  --weights 0.7 0.3 \
  --output_csv outputs/pseudo_blend_round1/pseudo_labels_round1_blended.csv \
  --output_json outputs/pseudo_blend_round1/pseudo_labels_round1_blended.json

python3 train_student.py \
  --metadata_csv data/train_metadata.csv \
  --labeled_window_manifest_csv birdclef_prepared/window_manifest.csv \
  --pseudo_label_csv outputs/pseudo_blend_round1/pseudo_labels_round1_blended.csv \
  --class_list_path birdclef_prepared/classes.txt \
  --output_dir outputs/birdclef_student_round1 \
  --use_mil

python3 export_int8.py \
  --student_checkpoint outputs/birdclef_student_round1/student_best.pth \
  --calibration_window_manifest_csv birdclef_prepared/window_manifest.csv \
  --output_dir outputs/birdclef_export

python3 infer_kaggle.py \
  --model_path outputs/birdclef_export/student.int8.onnx \
  --class_list_path birdclef_prepared/classes.txt \
  --test_csv data/test_metadata.csv \
  --sample_submission_csv data/sample_submission.csv \
  --output_csv submission.csv

python3 package_kaggle_artifacts.py \
  --model_path outputs/birdclef_export/student.int8.onnx \
  --class_list_path birdclef_prepared/classes.txt \
  --output_dir final_artifacts \
  --zip

python3 package_teacher_kaggle_artifacts.py \
  --teacher_manifest_json outputs/birdclef_teachers/teacher_manifest.json \
  --class_list_path birdclef_prepared/classes.txt \
  --output_dir teacher_kaggle_artifacts \
  --zip

python3 publish_kaggle_dataset.py \
  --dataset_dir teacher_kaggle_artifacts \
  --dataset_id YOUR_USERNAME/birdclef-teacher-artifacts \
  --title "BirdCLEF Teacher Artifacts" \
  --version_message "Teacher submission bundle"

python3 infer_teacher_kaggle.py \
  --teacher_manifest_json outputs/birdclef_teachers/teacher_manifest.json \
  --audio_dir /kaggle/input/competitions/birdclef-2026/test_soundscapes \
  --sample_submission_csv /kaggle/input/competitions/birdclef-2026/sample_submission.csv \
  --output_csv submission.csv

python3 compare_onnx_models.py \
  --fp32_model_path outputs/birdclef_export/student.onnx \
  --int8_model_path outputs/birdclef_export/student.int8.onnx \
  --class_list_path birdclef_prepared/classes.txt \
  --metadata_csv data/train_metadata.csv \
  --sample_n_files 128 \
  --output_json onnx_compare_report.json \
  --per_file_csv onnx_compare_per_file.csv

python3 compare_teacher_student_models.py \
  --teacher_manifest_json outputs/birdclef_teachers/teacher_manifest.json \
  --window_manifest_csv birdclef_prepared/window_manifest.csv \
  --metadata_csv data/train_metadata.csv \
  --class_list_path birdclef_prepared/classes.txt \
  --fp32_model_path outputs/birdclef_export/student.onnx \
  --int8_model_path outputs/birdclef_export/student.int8.onnx \
  --sample_n_files 128 \
  --output_json teacher_student_compare_report.json \
  --per_file_csv teacher_student_compare_per_file.csv

python3 kaggle_submission_runner.py \
  --artifact_dir /kaggle/input/YOUR_DATASET_NAME \
  --competition_dir /kaggle/input/competitions/birdclef-2026 \
  --wheel_path /kaggle/input/YOUR_WHEEL_DATASET/onnxruntime-*.whl

python3 teacher_kaggle_submission_runner.py \
  --artifact_dir /kaggle/input/YOUR_TEACHER_DATASET \
  --competition_dir /kaggle/input/competitions/birdclef-2026 \
  --max_teachers 3 \
  --device cpu
```

Stronger backbone examples:

```bash
python3 train_teachers.py \
  --metadata_csv data/train_metadata.csv \
  --window_manifest_csv birdclef_prepared/window_manifest.csv \
  --class_list_path birdclef_prepared/classes.txt \
  --output_dir outputs/birdclef_teachers_sota \
  --teacher_model_types efficientnet_v2_m convnext_small \
  --train_perch_teacher \
  --use_mil

python3 train_student.py \
  --metadata_csv data/train_metadata.csv \
  --labeled_window_manifest_csv birdclef_prepared/window_manifest.csv \
  --pseudo_label_csv outputs/pseudo_round1/pseudo_labels_round1.csv \
  --class_list_path birdclef_prepared/classes.txt \
  --output_dir outputs/birdclef_student_convnext \
  --backbone convnext_tiny \
  --use_mil

python3 train_perch_student.py \
  --metadata_csv data/train_metadata.csv \
  --labeled_window_manifest_csv birdclef_prepared/window_manifest.csv \
  --pseudo_label_csv outputs/pseudo_round1/pseudo_labels_round1.csv \
  --class_list_path birdclef_prepared/classes.txt \
  --output_dir outputs/birdclef_perch_student \
  --hidden_dim 768 \
  --num_heads 8 \
  --num_layers 4
```

Notes:
- `prepare_kaggle_data.py` downloads the competition with the Kaggle CLI, extracts the archives, writes `data/train_metadata.csv`, writes `data/test_metadata.csv`, and copies `data/sample_submission.csv`.
- The download step requires Kaggle API credentials to be configured for the current environment.
- Raw Kaggle metadata is normalized into the standalone contract: `soundscape_id`, `audio_path`, and `labels`.
- `infer_kaggle.py` uses `sample_submission.csv` as the authoritative row list when it is provided, so submission row IDs and column order match Kaggle exactly.
- Supported image backbones are `efficientnet_v2_s`, `efficientnet_v2_m`, `convnext_tiny`, and `convnext_small`.
- `--train_perch_teacher` requires separate Perch tooling and cached/window embedding support. If you enable it, run `preprocess.py` with `--compute_perch_embeddings`.
- `train_perch_student.py` also requires `preprocess.py --compute_perch_embeddings`; it trains a file-level transformer over cached Perch window embeddings.
- `blend_pseudolabels.py` averages class probabilities across pseudo-label CSVs and rebuilds a training-ready subset against the reference window manifest.
- Optional grouped CV columns are `author`, `recordist`, and `site`.
- Final submission inference requires real test audio. If your local environment only has `test_soundscapes/readme.txt`, build `final_artifacts` and run `infer_kaggle.py` inside a Kaggle notebook instead.
