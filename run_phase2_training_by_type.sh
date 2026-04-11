#!/bin/bash
# =============================================================================
# 第二阶段训练：按任务类型批量微调 (train_by_task_type.py)
# - 依赖第一阶段 run_phase1_unified_training.sh 产出的联合模型权重
# - 默认从 mit_b5_v2_best_model.pth 初始化，为四类任务各训一份 specialist
# - 若尚未完成第一阶段，请先: bash run_phase1_unified_training.sh
# =============================================================================

# 配置（须与第一阶段 train.py 的 MODEL_SAVE_PATH 一致）
PRETRAINED_MODEL="mit_b5_v2_best_model.pth"
DATA_ROOT="./data/train"
NUM_EPOCHS=10
BATCH_SIZE=10
LEARNING_RATE=1e-5
LOG_DIR="logs"
OUTPUT_DIR="task_type_models"

# 四个任务类型
TASK_TYPES=(
    "segmentation"    # 分割任务 (12个子任务)
    "classification"  # 分类任务 (9个子任务)
    "Regression"      # 回归任务 (3个子任务)
    "detection"       # 检测任务 (3个子任务)
)

if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "错误: 找不到第一阶段权重: $PRETRAINED_MODEL"
    echo "请先运行第一阶段: bash run_phase1_unified_training.sh"
    exit 1
fi

echo "=========================================="
echo "第二阶段：按任务类型批量微调"
echo "=========================================="
echo "预训练模型 (来自第一阶段): $PRETRAINED_MODEL"
echo "数据目录: $DATA_ROOT"
echo "训练轮数: $NUM_EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "学习率: $LEARNING_RATE"
echo "输出目录: $OUTPUT_DIR"
echo "日志目录: $LOG_DIR"
echo "任务类型数: ${#TASK_TYPES[@]}"
echo "=========================================="

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# 训练每个任务类型
for i in "${!TASK_TYPES[@]}"; do
    TASK_TYPE="${TASK_TYPES[$i]}"
    TASK_NUM=$((i + 1))
    
    echo ""
    echo "=========================================="
    echo "[$TASK_NUM/${#TASK_TYPES[@]}] [第二阶段] 开始训练任务类型: $TASK_TYPE"
    echo "=========================================="
    
    python train_by_task_type.py \
        --task_type $TASK_TYPE \
        --pretrained_model $PRETRAINED_MODEL \
        --data_root $DATA_ROOT \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --log_dir $LOG_DIR \
        --output_dir $OUTPUT_DIR \
        --encoder mit_b5 \
        --encoder_weights imagenet \
        --val_split 0.2 \
        --random_seed 42
    
    if [ $? -eq 0 ]; then
        echo "✓ 任务类型 $TASK_TYPE 训练完成"
    else
        echo "✗ 任务类型 $TASK_TYPE 训练失败"
    fi
done

echo ""
echo "=========================================="
echo "第二阶段：所有任务类型训练完成"
echo "=========================================="
echo "模型保存目录: $OUTPUT_DIR"
echo "日志保存目录: $LOG_DIR"

# 列出所有生成的模型文件
echo ""
echo "已生成的模型文件:"
ls -lh $OUTPUT_DIR/*.pth 2>/dev/null || echo "没有找到模型文件"

# 显示模型大小统计
echo ""
echo "模型文件大小统计:"
du -sh $OUTPUT_DIR 2>/dev/null || echo "输出目录不存在"
