# BirdCLEF Standalone

Standalone BirdCLEF-style multilabel soundscape pipeline extracted from this repo.

Files:
- `prepare_kaggle_data.py`
- `preprocess.py`
- `dataset.py`
- `train_teachers.py`
- `generate_pseudolabels.py`
- `train_student.py`
- `export_int8.py`
- `infer_kaggle.py`
- `kaggle_submission_runner.py`
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

python3 train_student.py \
  --metadata_csv data/train_metadata.csv \
  --labeled_window_manifest_csv birdclef_prepared/window_manifest.csv \
  --pseudo_label_csv outputs/pseudo_round1/pseudo_labels_round1.csv \
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

python3 kaggle_submission_runner.py \
  --artifact_dir /kaggle/input/YOUR_DATASET_NAME \
  --competition_dir /kaggle/input/competitions/birdclef-2026 \
  --wheel_path /kaggle/input/YOUR_WHEEL_DATASET/onnxruntime-*.whl
```

Notes:
- `prepare_kaggle_data.py` downloads the competition with the Kaggle CLI, extracts the archives, writes `data/train_metadata.csv`, writes `data/test_metadata.csv`, and copies `data/sample_submission.csv`.
- The download step requires Kaggle API credentials to be configured for the current environment.
- Raw Kaggle metadata is normalized into the standalone contract: `soundscape_id`, `audio_path`, and `labels`.
- `infer_kaggle.py` uses `sample_submission.csv` as the authoritative row list when it is provided, so submission row IDs and column order match Kaggle exactly.
- `--train_perch_teacher` requires separate Perch tooling and cached/window embedding support. If you enable it, run `preprocess.py` with `--compute_perch_embeddings`.
- Optional grouped CV columns are `author`, `recordist`, and `site`.
- Final submission inference requires real test audio. If your local environment only has `test_soundscapes/readme.txt`, build `final_artifacts` and run `infer_kaggle.py` inside a Kaggle notebook instead.
