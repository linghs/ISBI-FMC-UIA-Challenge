# Ultrasound Multi-Task Learning (MiT + Faster R-CNN)

[![ISBI 2026](https://img.shields.io/badge/ISBI-2026-red)](https://biomedicalimaging.org/)

**A Unified Multi-Task Learning Framework for Ultrasound Image Analysis**  
*Accepted at IEEE International Symposium on Biomedical Imaging (ISBI) 2026*

Shared-backbone multi-task training and task-type–specialized fine-tuning for **Foundation Model Challenge for Ultrasound Image Analysis (FMC_UIA)**–style benchmarks. Backbone: **MiT-B5** (`mit_b5`); detection heads use **Faster R-CNN**–style components where applicable.


**[中文说明 → README.zh.md](README.zh.md)**

---

## Overview

This repository implements a **two-stage** workflow:

1. **Phase 1 — unified multi-task training** (`train.py`): one model is trained jointly on all tasks (segmentation, classification, regression, detection) with a shared encoder. The best checkpoint is saved as `mit_b5_v2_best_model.pth` (configurable in `train.py`).
2. **Phase 2 — per–task-type fine-tuning** (`train_by_task_type.py`): four specialist checkpoints are trained (one per task type), initialized from the Phase 1 weights, and stored under `task_type_models/`.
3. **Inference** (`predict_by_task_type.py`): run predictions on a split (e.g. validation) using the Phase 2 checkpoints.

Convenience shell scripts: `run_phase1_unified_training.sh`, `run_phase2_training_by_type.sh`, and `predict_multi_task.sh`.

---

## Directory & path format

Training and inference both expect a **data root** (`--data_root` / `DATA_ROOT_PATH`), e.g. `./data/train` or `./data/val`.



### How paths in CSV are resolved

Code joins each row’s file fields with the **`csv_files` directory**, not with `data_root` directly:

- **Image file:**  
  `abs_path = os.path.join({data_root}/csv_files, row['image_path'])`

So **`image_path` must be relative to `{data_root}/csv_files/`** (use forward slashes; avoid leading `/` unless you intend an absolute path).  
Example: if the image file is at `{data_root}/csv_files/images/case001.png`, put `images/case001.png` in the `image_path` column.

- **Segmentation mask:** same rule for `mask_path` when present: **relative to `{data_root}/csv_files/`**.

### Required columns (all tasks)

| Column       | Description |
|-------------|-------------|
| `task_id`   | Task identifier (must match `model_factory.TASK_CONFIGURATIONS`, e.g. `fetal_plane_cls`). |
| `task_name` | One of `segmentation`, `classification`, `Regression`, `detection`. |
| `image_path`| Path to the image file, **relative to `csv_files/`**. |

### Task-specific columns

| `task_name`     | Extra columns |
|-----------------|---------------|
| `segmentation`  | `mask_path` (optional): mask image path relative to `csv_files/`. |
| `classification`| `mask`: integer class label (column name is historical). |
| `Regression`    | `num_classes`: number of landmarks; `point_1_xy`, `point_2_xy`, … each cell is a **JSON list** of coordinates in **pixel** space, e.g. `"[x, y]"` (see `dataset.py`). |
| `detection`     | `x_min`, `y_min`, `x_max`, `y_max`: bounding box in **pixels** (training); inference CSV still supplies metadata consistent with the challenge format. |

Paths use `os.path.normpath` after joining; prefer **POSIX-style** relative paths inside CSV for cross-platform use.

---

## Requirements

- Python 3.10+ recommended  
- CUDA-capable GPU recommended for training  

```bash
pip install -r requirements.txt
```

---

## Usage

**Phase 1** — edit hyperparameters at the top of `train.py` (`LEARNING_RATE`, `BATCH_SIZE`, `NUM_EPOCHS`, `DATA_ROOT_PATH`, `MODEL_SAVE_PATH`, …):

```bash
bash run_phase1_unified_training.sh
```

**Phase 2** — requires `mit_b5_v2_best_model.pth` from Phase 1; adjust `run_phase2_training_by_type.sh` if needed:

```bash
bash run_phase2_training_by_type.sh
```

**Prediction** — defaults: `./data/val`, models `./task_type_models`, output `./predictions_multi_task` (edit `predict_multi_task.sh`):

```bash
bash predict_multi_task.sh
```

---

## Project structure (core files)

| File | Role |
|------|------|
| `train.py` | Phase 1 joint training |
| `train_by_task_type.py` | Phase 2 training by task type |
| `predict_by_task_type.py` | Inference with task-type checkpoints |
| `dataset.py` | `MultiTaskDataset` and sampling |
| `model_factory.py` | Model factory and `TASK_CONFIGURATIONS` |
| `utils.py` | Losses, metrics, collate, evaluation helpers |
| `requirements.txt` | Python dependencies |

---

## Outputs

- Phase 1: `mit_b5_v2_best_model.pth` (default)  
- Phase 2: `task_type_models/*.pth`  
- Prediction: JSON and masks under `--output_dir` (see `predict_by_task_type.py`)

---

## License

Add a `LICENSE` file if you publish this repository publicly.
