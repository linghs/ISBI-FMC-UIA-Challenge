#!/bin/bash
# =============================================================================
# 第一阶段训练：多任务联合训练 (train.py)
# - 使用 MultiTaskDataset 在所有任务上联合优化一个共享骨干模型
# - 产出权重: mit_b5_v2_best_model.pth（路径与 train.py 中 MODEL_SAVE_PATH 一致）
# - 超参请在 train.py 顶部修改（LR/BATCH/EPOCHS/DATA_ROOT_PATH 等）
# =============================================================================

set -e

PHASE1_OUTPUT="mit_b5_v2_best_model.pth"
LOG_DIR="logs"

echo "=========================================="
echo "第一阶段：联合多任务训练 (train.py)"
echo "=========================================="
echo "说明: 本阶段不接收命令行超参，请编辑 train.py 内配置"
echo "数据与保存路径以 train.py 为准，默认输出: ${PHASE1_OUTPUT}"
echo "日志目录: ${LOG_DIR}"
echo "=========================================="
echo ""

mkdir -p "${LOG_DIR}"

python train.py

echo ""
echo "=========================================="
echo "第一阶段训练结束"
echo "=========================================="
if [ -f "${PHASE1_OUTPUT}" ]; then
    echo "✓ 已生成: ${PHASE1_OUTPUT}"
    ls -lh "${PHASE1_OUTPUT}"
else
    echo "⚠ 未找到 ${PHASE1_OUTPUT}，请检查 train.py 是否成功保存或路径是否被修改"
    exit 1
fi
echo ""
echo "下一步: 运行第二阶段按任务类型微调"
echo "  bash run_phase2_training_by_type.sh"
echo "=========================================="
