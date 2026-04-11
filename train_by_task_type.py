#!/usr/bin/env python3
"""
按任务类型训练脚本
将相同类型的任务（分割/分类/回归/检测）一起训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch.losses as smp_losses
import numpy as np
import logging
import os
import argparse
from datetime import datetime

# Import local modules
from dataset import MultiTaskDataset, MultiTaskUniformSampler
from model_factory import MultiTaskModelFactory, TASK_CONFIGURATIONS
from utils import (
    multi_task_collate_fn, 
    evaluate, 
    DetectionLoss, 
    FasterRCNNLoss,
    set_seed
)


def setup_logger(log_dir='logs', task_type='task_type'):
    """设置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{task_type}_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件已创建: {log_file}")
    return logger, log_file


def get_tasks_by_type(task_type):
    """根据任务类型获取所有相关的任务ID"""
    tasks = []
    for cfg in TASK_CONFIGURATIONS:
        if cfg['task_name'] == task_type:
            tasks.append(cfg['task_id'])
    return tasks


def filter_dataset_by_task_type(dataset, task_ids):
    """筛选出特定任务类型的所有数据"""
    task_indices = []
    for idx in range(len(dataset)):
        if dataset.dataframe.iloc[idx]['task_id'] in task_ids:
            task_indices.append(idx)
    return task_indices


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='按任务类型训练脚本')
    parser.add_argument('--task_type', type=str, required=True,
                        choices=['segmentation', 'classification', 'Regression', 'detection'],
                        help='要训练的任务类型: segmentation/classification/Regression/detection')
    parser.add_argument('--pretrained_model', type=str, 
                        help='预训练多任务模型的路径')
    parser.add_argument('--data_root', type=str, default='./data/train',
                        help='数据根目录')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='基础学习率')
    parser.add_argument('--encoder', type=str, default='mit_b5',
                        help='编码器名称')
    parser.add_argument('--encoder_weights', type=str, default='imagenet',
                        help='编码器预训练权重')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='验证集比例')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='是否冻结编码器')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='日志目录')
    parser.add_argument('--output_dir', type=str, default='task_type_models',
                        help='模型输出目录')
    parser.add_argument('--detection_label_perturbation', action='store_true',
                        help='对检测任务启用标签扰动（用于域适应，论文方法）')
    parser.add_argument('--regression_label_perturbation', action='store_true',
                        help='对回归任务启用标签扰动（用于域适应，论文方法）')
    parser.add_argument('--perturbation_sigma', type=float, default=2.0,
                        help='标签扰动的高斯噪声标准差（像素），默认2.0')
    
    args = parser.parse_args()
    
    # 获取该类型的所有任务
    task_ids = get_tasks_by_type(args.task_type)
    
    if not task_ids:
        print(f"错误: 任务类型 '{args.task_type}' 没有找到任何任务")
        return
    
    # 设置日志
    logger, log_file = setup_logger(args.log_dir, args.task_type)
    
    # 设置随机种子
    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 记录训练配置
    logger.info("="*70)
    logger.info(f"按任务类型训练 - 任务类型: {args.task_type}")
    logger.info("="*70)
    logger.info(f"包含的任务 ({len(task_ids)}个):")
    for task_id in task_ids:
        task_cfg = next((cfg for cfg in TASK_CONFIGURATIONS if cfg['task_id'] == task_id), None)
        if task_cfg:
            logger.info(f"  - {task_id:<30} (类别数: {task_cfg['num_classes']})")
    logger.info(f"使用设备: {device}")
    logger.info(f"预训练模型: {args.pretrained_model}")
    logger.info(f"数据根目录: {args.data_root}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"训练轮数: {args.num_epochs}")
    logger.info(f"学习率: {args.learning_rate}")
    logger.info(f"编码器: {args.encoder} (预训练权重: {args.encoder_weights})")
    logger.info(f"验证集比例: {args.val_split}")
    logger.info(f"随机种子: {args.random_seed}")
    logger.info(f"冻结编码器: {args.freeze_encoder}")
    if args.task_type == 'detection' and args.detection_label_perturbation:
        logger.info(f"检测标签扰动: 启用 (σ={args.perturbation_sigma} pixels)")
    if args.task_type == 'Regression' and args.regression_label_perturbation:
        logger.info(f"回归标签扰动: 启用 (σ={args.perturbation_sigma} pixels) - Device-Domain Adaptation")
    logger.info("="*70)
    
    # 数据增强配置
    # train_transforms = A.Compose([
    #     A.Resize(256, 256), 
    #     A.RandomBrightnessContrast(p=0.2),
    #     A.GaussNoise(p=0.1), 
    #     A.HorizontalFlip(p=0.5),
    #     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #     ToTensorV2(),
    # ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))
    
    # val_transforms = A.Compose([
    #     A.Resize(256, 256),
    #     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #     ToTensorV2(),
    # ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))
    
    train_transforms = A.Compose([
        A.Resize(512, 512), 
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.1), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))
    
    # Validation transforms without augmentation
    val_transforms = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))

    # 创建数据集
    # 对检测任务启用标签扰动（如果指定）
    # enable_perturbation = (args.task_type == 'detection' and args.detection_label_perturbation)
    
    full_dataset = MultiTaskDataset(
        data_root=args.data_root, 
        transforms=train_transforms,
        regression_label_perturbation=False,   # 对回归任务启用标签扰动
        perturbation_sigma=args.perturbation_sigma
    )
    
    # 筛选该类型的所有任务数据
    type_indices = filter_dataset_by_task_type(full_dataset, task_ids)
    logger.info(f"任务类型 {args.task_type} 共有 {len(type_indices)} 个样本")
    
    # 验证筛选结果
    logger.info(f"验证筛选结果（检查前10个样本）:")
    for i, idx in enumerate(type_indices[:10]):
        task_id = full_dataset.dataframe.iloc[idx]['task_id']
        logger.info(f"  样本 {i}: 索引={idx}, task_id={task_id}")
    # 统计每个任务的样本数
    task_sample_counts = defaultdict(int)
    for idx in type_indices:
        task_id = full_dataset.dataframe.iloc[idx]['task_id']
        task_sample_counts[task_id] += 1
    
    logger.info("各任务样本分布:")
    for task_id in sorted(task_sample_counts.keys()):
        logger.info(f"  - {task_id:<30}: {task_sample_counts[task_id]} 样本")
    
    if len(type_indices) == 0:
        logger.error(f"错误: 任务类型 {args.task_type} 没有找到任何数据!")
        return
    
    # 划分训练集和验证集
    val_size = int(len(type_indices) * args.val_split)
    train_size = len(type_indices) - val_size
    
    generator = torch.Generator().manual_seed(args.random_seed)
    train_indices_split, val_indices_split = torch.utils.data.random_split(
        type_indices, [train_size, val_size], generator=generator
    )
    
    # 获取实际的索引（type_indices中的索引，而不是split后的索引）
    actual_train_indices = [type_indices[i] for i in train_indices_split.indices]
    actual_val_indices = [type_indices[i] for i in val_indices_split.indices]
    
    logger.info(f"数据集划分: 训练集 {train_size} 样本, 验证集 {val_size} 样本")
    logger.info(f"验证训练集前5个样本的task_id:")
    for i, idx in enumerate(actual_train_indices[:5]):
        task_id = full_dataset.dataframe.iloc[idx]['task_id']
        logger.info(f"  样本 {i}: 索引={idx}, task_id={task_id}")
    
    # 创建训练集和验证集
    train_dataset = MultiTaskDataset(
        data_root=args.data_root, 
        transforms=train_transforms,
        regression_label_perturbation=True,   # 对回归任务启用标签扰动
        perturbation_sigma=args.perturbation_sigma
    )
    val_dataset = MultiTaskDataset(
        data_root=args.data_root, 
        transforms=val_transforms,
        regression_label_perturbation=False,
        perturbation_sigma=args.perturbation_sigma
    )
    
    train_subset = torch.utils.data.Subset(train_dataset, actual_train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, actual_val_indices)
    
    # Fix dataframe reference for subset (for sampler)
    train_subset.dataframe = train_dataset.dataframe.iloc[actual_train_indices].reset_index(drop=True)
    
    logger.info(f"验证train_subset.dataframe的前5个task_id:")
    for i in range(min(5, len(train_subset.dataframe))):
        task_id = train_subset.dataframe.iloc[i]['task_id']
        logger.info(f"  行 {i}: task_id={task_id}")
    
    # 使用均衡采样器（确保每个任务被均匀采样）
    train_sampler = MultiTaskUniformSampler(train_subset, batch_size=args.batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=multi_task_collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=multi_task_collate_fn
    )
    
    # 创建模型
    logger.info("初始化模型...")
    model = MultiTaskModelFactory(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        task_configs=TASK_CONFIGURATIONS
    ).to(device)
    
    # 加载预训练权重
    if args.pretrained_model is not None and args.pretrained_model != '' and os.path.exists(args.pretrained_model):
        logger.info(f"加载预训练权重: {args.pretrained_model}")
        try:
            state_dict = torch.load(args.pretrained_model, map_location=device)
            model.load_state_dict(state_dict)
            logger.info("预训练权重加载成功!")
        except Exception as e:
            logger.warning(f"加载预训练权重时出现警告: {e}")
            logger.warning("将继续使用随机初始化的权重...")
    else:
        logger.warning(f"预训练模型文件不存在: {args.pretrained_model}")
        logger.warning("将使用随机初始化的权重...")
    
    # 冻结编码器(如果需要)
    if args.freeze_encoder:
        logger.info("冻结编码器参数...")
        for param in model.encoder.parameters():
            param.requires_grad = False
    
    # 设置损失函数
    loss_functions = {
        'segmentation': smp_losses.DiceLoss(mode='multiclass'),
        'classification': nn.CrossEntropyLoss(),
        'Regression': nn.MSELoss(),
        'detection': FasterRCNNLoss()  # Use FasterRCNNLoss for Faster R-CNN
    }
    
    criterion = loss_functions[args.task_type]
    logger.info(f"损失函数: {criterion.__class__.__name__}")
    
    # 构建任务ID到任务名称的映射
    task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in TASK_CONFIGURATIONS}
    
    # 设置优化器 - 只优化相关的任务头
    logger.info("设置优化器参数组...")
    param_groups = []
    
    if not args.freeze_encoder:
        param_groups.append({
            'params': model.encoder.parameters(), 
            'lr': args.learning_rate * 0.1
        })
        logger.info(f"  - 编码器: LR = {args.learning_rate * 0.1:.2e}")
    
    # 如果是分割或检测任务，需要训练FPN decoder
    if args.task_type in ['segmentation', 'detection']:
        param_groups.append({
            'params': model.fpn_decoder.parameters(),
            'lr': args.learning_rate * 0.5
        })
        logger.info(f"  - FPN解码器: LR = {args.learning_rate * 0.5:.2e}")
    
    # 添加所有相关任务头的参数
    for task_id in task_ids:
        if task_id in model.heads:
            head = model.heads[task_id]
            
            # For detection tasks with Faster R-CNN, we need to handle parameters carefully
            # to avoid duplicate parameters (encoder and fpn_decoder are already added)
            if task_id_to_name[task_id] == 'detection':
                # Only add Faster R-CNN specific parameters (RPN, ROI head, etc.)
                # Exclude backbone (encoder + fpn_decoder) parameters
                
                # Get all parameters from the Faster R-CNN model
                faster_rcnn_params = set(head.model.parameters())
                
                # Get backbone parameters to exclude
                backbone_params = set(head.model.backbone.parameters())
                
                # Only add non-backbone parameters
                specific_params = list(faster_rcnn_params - backbone_params)
                
                if specific_params:
                    param_groups.append({
                        'params': specific_params,
                        'lr': args.learning_rate
                    })
                    logger.info(f"  - 任务头 '{task_id}' (Faster R-CNN特定参数): LR = {args.learning_rate:.2e}")
            else:
                # For other task types, add all parameters
                param_groups.append({
                    'params': head.parameters(),
                    'lr': args.learning_rate
                })
                logger.info(f"  - 任务头 '{task_id}': LR = {args.learning_rate:.2e}")
    
    optimizer = optim.AdamW(param_groups)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    logger.info("学习率调度器: CosineAnnealingLR")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    model_save_path = os.path.join(args.output_dir, f'{args.task_type}_best_model.pth')
    
    # 训练循环
    best_val_score = -float('inf') if args.task_type != 'Regression' else float('inf')
    logger.info("="*70)
    logger.info("开始训练")
    logger.info("="*70)
    
    for epoch in range(args.num_epochs):
        # ===== 训练阶段 =====
        model.train()
        epoch_train_losses = defaultdict(list)
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        
        for batch in loop:
            images = batch['image'].to(device)
            task_ids_batch = batch['task_id']
            labels = torch.stack(batch['label']).to(device)
            
            # 当前batch的任务ID（由于使用了均衡采样器，一个batch只有一个任务）
            current_task_id = task_ids_batch[0]
            task_name = task_id_to_name[current_task_id]
            
            # 前向传播
            # Detection任务需要特殊处理（Faster R-CNN）
            if task_name == 'detection':
                # Prepare targets in Faster R-CNN format
                targets = []
                for i in range(images.shape[0]):
                    # labels is [x1, y1, x2, y2] in normalized coordinates
                    # Convert to absolute coordinates (assuming 256x256 image)
                    boxes = labels[i:i+1] * 512  # Denormalize
                    
                    target = {
                        'boxes': boxes.float(),  # [N, 4] format: [x1, y1, x2, y2]
                        'labels': torch.ones((1,), dtype=torch.int64, device=device)  # All class 1
                    }
                    targets.append(target)
                
                # Get detection head and forward pass
                detection_head = model.heads[current_task_id]
                
                # Convert images to list format (required by Faster R-CNN)
                image_list = [images[i] for i in range(images.shape[0])]
                
                # Forward pass returns loss dict in training mode
                loss_dict = detection_head(image_list, targets)
                
                # Calculate total loss
                loss = criterion(loss_dict)
            else:
                outputs = model(images, task_id=current_task_id)
                
                # 计算损失
                loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_losses[current_task_id].append(loss.item())
            loop.set_postfix(loss=loss.item(), task=current_task_id, lr=scheduler.get_last_lr()[0])
        
        # 训练损失报告
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs} - 训练损失汇总")
        logger.info(f"{'='*70}")
        total_loss = 0
        total_samples = 0
        for task_id in sorted(epoch_train_losses.keys()):
            avg_loss = np.mean(epoch_train_losses[task_id])
            num_samples = len(epoch_train_losses[task_id])
            total_loss += avg_loss * num_samples
            total_samples += num_samples
            logger.info(f"  - 任务 '{task_id:<30}': 平均Loss = {avg_loss:.6f} ({num_samples} batches)")
        
        overall_avg_loss = total_loss / total_samples if total_samples > 0 else 0
        logger.info(f"  - 总体平均Loss: {overall_avg_loss:.6f}")
        logger.info("-"*70)
        
        # ===== 验证阶段 =====
        val_results_df = evaluate(model, val_loader, device)
        
        # 计算验证分数
        if not val_results_df.empty:
            logger.info(f"\n{'='*70}")
            logger.info(f"Epoch {epoch + 1}/{args.num_epochs} - 验证集评估结果")
            logger.info(f"{'='*70}")
            
            # 只显示当前任务类型的结果
            type_results = val_results_df[val_results_df['Task ID'].isin(task_ids)]
            metrics = []
            for _, row in type_results.iterrows():
                for col in type_results.columns:
                    if col != 'Task ID' and col != 'Task Name' and isinstance(row[col], (int, float)):
                        metrics.append(f"{row[col]:.4f}")
                        logger.info(f"  任务 '{row['Task ID']:<30}': {col}={row[col]:.4f}")
            
            # 根据任务类型选择评估指标
            if args.task_type == 'Regression':
                # Convert metrics to float before calculating mean to avoid dtype errors
                metrics_float = [float(m) for m in metrics]
                current_val_score = np.mean(metrics_float) if metrics_float else float('inf')
                is_better = current_val_score < best_val_score
                # 回归任务: MAE越小越好
            else:
                metrics_float = [float(m) for m in metrics]
                current_val_score = np.mean(metrics_float) if metrics_float else float('inf')
                is_better = current_val_score > best_val_score
                
            # 保存最佳模型
            if is_better:
                best_val_score = current_val_score
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"🎉 新的最佳模型已保存! 验证分数: {best_val_score:.6f}")
                logger.info(f"   模型保存路径: {model_save_path}")
            else:
                logger.info(f"   验证分数未提升 (当前: {current_val_score:.6f}, 最佳: {best_val_score:.6f})")
        # 更新学习率
        scheduler.step()
        logger.info(f"   当前学习率: {scheduler.get_last_lr()[0]:.8f}")
        logger.info("-"*70)
    
    # 训练完成
    logger.info("="*70)
    logger.info("训练完成!")
    logger.info("="*70)
    logger.info(f"任务类型: {args.task_type}")
    logger.info(f"包含任务: {', '.join(task_ids)}")
    logger.info(f"最佳验证分数: {best_val_score:.6f}")
    logger.info(f"最佳模型保存路径: {model_save_path}")
    logger.info(f"日志文件: {log_file}")
    logger.info("="*70)
    
    print(f"\n训练完成!")
    print(f"最佳模型已保存至: {model_save_path}")
    print(f"日志文件: {log_file}")


if __name__ == '__main__':
    main()

