# 超声多任务学习（MiT + Faster R-CNN）

面向 **Foundation Model Challenge for Ultrasound Image Analysis (FMC_UIA)** 类赛题的共享骨干多任务训练与按任务类型微调。骨干网络为 **MiT-B5**（`mit_b5`）；检测分支在适用场景下采用 **Faster R-CNN** 风格结构。

**[English README → README.md](README.md)**

---

## 概述

本仓库实现**两阶段**流程：

1. **第一阶段：联合多任务训练**（`train.py`）— 在分割、分类、回归、检测上共享编码器联合训练，验证集最优权重默认保存为 `mit_b5_v2_best_model.pth`（可在 `train.py` 修改 `MODEL_SAVE_PATH`）。
2. **第二阶段：按任务类型微调**（`train_by_task_type.py`）— 以第一阶段权重初始化，对四类任务类型分别训练 specialist，默认输出到 `task_type_models/`。
3. **预测**（`predict_by_task_type.py`）— 使用第二阶段权重对验证集等进行推理。

便捷脚本：`run_phase1_unified_training.sh`、`run_phase2_training_by_type.sh`、`predict_multi_task.sh`。

---

## 目录与路径格式

训练与推理均使用**数据根目录**（`--data_root` / `DATA_ROOT_PATH`），例如 `./data/train` 或 `./data/val`。



### CSV 中路径如何拼接

代码将表中的路径与 **`csv_files` 目录**拼接，而不是直接与 `data_root` 拼接：

- **图像：**  
  `实际路径 = {data_root}/csv_files/` + `image_path` 列

因此 **`image_path` 必须相对于 `{data_root}/csv_files/`** 书写（建议用正斜杠 `/`；除非故意使用绝对路径，否则不要以 `/` 开头）。  
例如文件在 `{data_root}/csv_files/images/case001.png`，则 `image_path` 填 `images/case001.png`。

- **分割任务 mask：** `mask_path` 同样**相对于 `{data_root}/csv_files/`**。

### 通用列（所有任务）

| 列名         | 说明 |
|--------------|------|
| `task_id`    | 子任务 ID，需与 `model_factory.TASK_CONFIGURATIONS` 中一致（如 `fetal_plane_cls`）。 |
| `task_name`  | `segmentation` / `classification` / `Regression` / `detection` 之一。 |
| `image_path` | 图像路径，**相对于 `csv_files/`**。 |

### 各任务类型额外列

| `task_name`     | 额外列 |
|-----------------|--------|
| `segmentation`  | `mask_path`（可选）：mask 相对 `csv_files/` 的路径。 |
| `classification`| `mask`：整数类别标签（列名沿用历史命名）。 |
| `Regression`    | `num_classes`：关键点数量；`point_1_xy`、`point_2_xy`、… 每项为 **JSON 列表**，像素坐标，如 `"[x, y]"`（见 `dataset.py`）。 |
| `detection`     | `x_min`, `y_min`, `x_max`, `y_max`：训练时为**像素坐标**的框。 |

拼接后会经 `os.path.normpath` 规范化；CSV 内建议使用**相对路径、正斜杠**，便于跨平台。

---

## 环境

```bash
pip install -r requirements.txt
```

建议 Python 3.10+；训练建议使用 NVIDIA GPU。

---

## 使用流程

**第一阶段**（超参在 `train.py` 顶部修改）：

```bash
bash run_phase1_unified_training.sh
```

**第二阶段**（需已有 `mit_b5_v2_best_model.pth`；可在 `run_phase2_training_by_type.sh` 中改 batch、学习率等）：

```bash
bash run_phase2_training_by_type.sh
```

**预测**（默认 `./data/val`、`./task_type_models`、输出 `./predictions_multi_task`，可改 `predict_multi_task.sh`）：

```bash
bash predict_multi_task.sh
```

---

## 核心文件

| 文件 | 说明 |
|------|------|
| `train.py` | 第一阶段联合训练 |
| `train_by_task_type.py` | 第二阶段按任务类型训练 |
| `predict_by_task_type.py` | 按任务类型推理 |
| `dataset.py` | 数据集与采样 |
| `model_factory.py` | 模型与任务配置 |
| `utils.py` | 损失、指标、评估 |
| `requirements.txt` | 依赖 |

---

## 产出物

- 第一阶段：默认 `mit_b5_v2_best_model.pth`  
- 第二阶段：`task_type_models/` 下各类型权重  
- 预测：在指定 `--output_dir` 下生成 JSON、分割结果等（见 `predict_by_task_type.py`）

---

## 许可证

公开仓库时请自行添加 `LICENSE` 并在本说明中更新。
