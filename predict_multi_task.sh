#!/bin/bash

# ============================================================================
# 多任务类型预测脚本
# 使用按任务类型训练的模型进行预测
# ============================================================================

# 设置错误时退出
set -e

echo "======================================================================"
echo "多任务类型预测"
echo "======================================================================"

# ============================================================================
# 配置参数
# ============================================================================

# 数据根目录
DATA_ROOT="./data/val"

# 输出目录
OUTPUT_DIR="./predictions_multi_task"

# 模型目录（包含按任务类型训练的模型）
MODEL_DIR="task_type_models"

# 批次大小
BATCH_SIZE=1

# 编码器名称
ENCODER="mit_b5"

# ============================================================================
# 选项1: 使用所有任务类型的模型进行预测
# ============================================================================
echo ""
echo "========================================================================"
echo "选项1: 使用所有任务类型模型进行预测"
echo "========================================================================"
echo "数据目录: ${DATA_ROOT}"
echo "输出目录: ${OUTPUT_DIR}"
echo "模型目录: ${MODEL_DIR}"
echo "批次大小: ${BATCH_SIZE}"
echo "========================================================================"
echo ""

python predict_by_task_type.py \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --model_dir ${MODEL_DIR} \
    --task_types  segmentation classification Regression detection \
    --batch_size ${BATCH_SIZE} \
    --encoder ${ENCODER}

echo ""
echo "✅ 所有任务类型预测完成!"
echo "结果保存在: ${OUTPUT_DIR}"
echo ""

# ============================================================================
# 选项2: 只使用部分任务类型（可按需取消注释）
# ============================================================================

# 例如：只预测分割和分类任务
# echo ""
# echo "========================================================================"
# echo "选项2: 只使用分割和分类模型进行预测"
# echo "========================================================================"
# OUTPUT_DIR_PARTIAL="./predictions_seg_cls"
# echo "输出目录: ${OUTPUT_DIR_PARTIAL}"
# echo "========================================================================"
# echo ""
# 
# python predict_by_task_type.py \
#     --data_root ${DATA_ROOT} \
#     --output_dir ${OUTPUT_DIR_PARTIAL} \
#     --model_dir ${MODEL_DIR} \
#     --task_types segmentation classification \
#     --batch_size ${BATCH_SIZE} \
#     --encoder ${ENCODER}
# 
# echo ""
# echo "✅ 分割和分类任务预测完成!"
# echo "结果保存在: ${OUTPUT_DIR_PARTIAL}"
# echo ""

# ============================================================================
# 选项3: 单独预测每个任务类型（调试用）
# ============================================================================

# # 分割任务
# echo ""
# echo "========================================================================"
# echo "预测: 分割任务"
# echo "========================================================================"
# python predict_by_task_type.py \
#     --data_root ${DATA_ROOT} \
#     --output_dir ./predictions_segmentation \
#     --model_dir ${MODEL_DIR} \
#     --task_types segmentation \
#     --batch_size ${BATCH_SIZE} \
#     --encoder ${ENCODER}
# echo "✅ 分割任务预测完成!"
# echo ""

# # 分类任务
# echo ""
# echo "========================================================================"
# echo "预测: 分类任务"
# echo "========================================================================"
# python predict_by_task_type.py \
#     --data_root ${DATA_ROOT} \
#     --output_dir ./predictions_classification \
#     --model_dir ${MODEL_DIR} \
#     --task_types classification \
#     --batch_size ${BATCH_SIZE} \
#     --encoder ${ENCODER}
# echo "✅ 分类任务预测完成!"
# echo ""

# # 回归任务
# echo ""
# echo "========================================================================"
# echo "预测: 回归任务"
# echo "========================================================================"
# python predict_by_task_type.py \
#     --data_root ${DATA_ROOT} \
#     --output_dir ./predictions_regression \
#     --model_dir ${MODEL_DIR} \
#     --task_types Regression \
#     --batch_size ${BATCH_SIZE} \
#     --encoder ${ENCODER}
# echo "✅ 回归任务预测完成!"
# echo ""

# # 检测任务
# echo ""
# echo "========================================================================"
# echo "预测: 检测任务"
# echo "========================================================================"
# python predict_by_task_type.py \
#     --data_root ${DATA_ROOT} \
#     --output_dir ./predictions_detection \
#     --model_dir ${MODEL_DIR} \
#     --task_types detection \
#     --batch_size ${BATCH_SIZE} \
#     --encoder ${ENCODER}
# echo "✅ 检测任务预测完成!"
# echo ""

# ============================================================================
# 完成
# ============================================================================
echo ""
echo "======================================================================"
echo "🎉 所有预测任务完成!"
echo "======================================================================"
echo "主要输出目录: ${OUTPUT_DIR}"
echo ""
echo "输出文件:"
echo "  - 分割结果: mask图像文件"
echo "  - 分类结果: ${OUTPUT_DIR}/classification_predictions.json"
echo "  - 检测结果: ${OUTPUT_DIR}/detection_predictions.json"
echo "  - 回归结果: ${OUTPUT_DIR}/regression_predictions.json"
echo "======================================================================"
echo ""

