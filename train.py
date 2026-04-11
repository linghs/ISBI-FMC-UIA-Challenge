import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch.losses as smp_losses
import numpy as np
import random
import logging
import os
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

# Training configuration
LEARNING_RATE = 1e-4
BATCH_SIZE = 10
NUM_EPOCHS = 10 
DATA_ROOT_PATH = './data/train'
ENCODER = 'mit_b5'
ENCODER_WEIGHTS = 'imagenet'
RANDOM_SEED = 42
MODEL_SAVE_PATH = 'mit_b5_v2_best_model.pth' 
VAL_SPLIT = 0.2
LOG_DIR = 'logs'  # 日志目录

def setup_logger(log_dir='logs'):
    """设置日志记录器"""
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件已创建: {log_file}")
    return logger, log_file

def main():
    # 设置日志记录器
    logger, log_file = setup_logger(LOG_DIR)
    
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")
    logger.info(f"使用设备: {device}")
    logger.info(f"随机种子: {RANDOM_SEED}")
    logger.info(f"训练配置: LR={LEARNING_RATE}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={NUM_EPOCHS}")
    logger.info(f"模型编码器: {ENCODER} (预训练权重: {ENCODER_WEIGHTS})")
    logger.info(f"验证集比例: {VAL_SPLIT}")

    # Data loading and splitting
    # Training transforms with augmentation
    train_transforms = A.Compose([
        A.Resize(512, 512), 
        # A.Resize(256, 256), 
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.1), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))
    
    # Validation transforms without augmentation
    val_transforms = A.Compose([
        A.Resize(512, 512), 
        # A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))

    # Create full dataset to get indices
    temp_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH,
                          transforms=train_transforms,
                          regression_label_perturbation=False)
    dataset_size = len(temp_dataset)
    val_size = int(dataset_size * VAL_SPLIT)
    train_size = dataset_size - val_size
    
    # Split indices
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = list(range(dataset_size))
    train_indices, val_indices = torch.utils.data.random_split(indices, [train_size, val_size], generator=generator)
    
    # Create separate datasets with different transforms
    train_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH,
                            transforms=train_transforms,
                            regression_label_perturbation=True) # 对回归任务启用标签扰动
    val_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH,
                            transforms=val_transforms,
                            regression_label_perturbation=False) # 对回归任务禁用标签扰动
    
    # Create subsets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    logger.info(f"数据集划分: 训练集 {train_size} 样本, 验证集 {val_size} 样本")
    
    # Fix dataframe reference for subset
    train_subset.dataframe = train_dataset.dataframe.iloc[train_indices.indices].reset_index(drop=True)
    
    train_sampler = MultiTaskUniformSampler(train_subset, batch_size=BATCH_SIZE)
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
    
    # Model and loss setup
    model = MultiTaskModelFactory(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, task_configs=TASK_CONFIGURATIONS).to(device)
    
    loss_functions = {
        'segmentation': smp_losses.DiceLoss(mode='multiclass'), 
        'classification': nn.CrossEntropyLoss(),
        'Regression': nn.MSELoss(), 
        'detection': FasterRCNNLoss()  # Use FasterRCNNLoss for Faster R-CNN
    }
    task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in TASK_CONFIGURATIONS}

    # Optimization setup
    print("\n--- Setting parameter groups ---")
    param_groups = [
        {'params': model.encoder.parameters(), 'lr': LEARNING_RATE * 1},
    ]
    print(f"  - Shared Encoder                 -> LR: {LEARNING_RATE * 1}")
    
    for task_id, head in model.heads.items():
        task_name = task_id_to_name[task_id]
        lr_multiplier = 10.0
        current_lr = LEARNING_RATE * lr_multiplier
        
        # For detection tasks, avoid parameter duplication
        if task_name == 'detection':
            # Only add Faster R-CNN specific parameters (RPN, ROI head, etc.)
            # Exclude backbone (encoder + fpn_decoder) parameters
            faster_rcnn_params = set(head.model.parameters())
            backbone_params = set(head.model.backbone.parameters())
            specific_params = list(faster_rcnn_params - backbone_params)
            
            if specific_params:
                param_groups.append({'params': specific_params, 'lr': current_lr})
                print(f"  - Task Head '{task_id:<25}' (Faster R-CNN) -> LR: {current_lr}")
        else:
            param_groups.append({'params': head.parameters(), 'lr': current_lr})
            print(f"  - Task Head '{task_id:<25}' -> LR: {current_lr}")

    optimizer = optim.AdamW(param_groups)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    print("\n--- Cosine Annealing Scheduler configured ---")

    best_val_score = -float('inf')
    print("\n" + "="*50 + "\n--- Start Training ---")
    logger.info("="*70)
    logger.info("开始训练")
    logger.info("="*70)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_losses = defaultdict(list)
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for batch in loop:
            images = batch['image'].to(device)
            task_ids = batch['task_id']
            # Manually stack labels list to tensor
            labels = torch.stack(batch['label']).to(device)

            # All samples in batch belong to the same task due to sampler
            current_task_id = task_ids[0]
            task_name = task_id_to_name[current_task_id]

            # Detection tasks need special handling (Faster R-CNN)
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
                loss = loss_functions[task_name](loss_dict)
            else:
                # Other task types use normal forward pass
                outputs = model(images, task_id=current_task_id)
                loss = loss_functions[task_name](outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_losses[current_task_id].append(loss.item())
            loop.set_postfix(loss=loss.item(), task=current_task_id, lr=scheduler.get_last_lr()[0])

        # Train reporting
        print("\n--- Epoch {} Average Train Loss Report ---".format(epoch + 1))
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS} - 训练损失汇总")
        logger.info(f"{'='*70}")
        sorted_task_ids = sorted(epoch_train_losses.keys())
        for task_id in sorted_task_ids:
            avg_loss = np.mean(epoch_train_losses[task_id])
            print(f"  - Task '{task_id:<25}': {avg_loss:.4f}")
            logger.info(f"任务 '{task_id:<25}': 平均Loss = {avg_loss:.6f}")
        print("-" * 40)
        logger.info("-"*70)

        # Validation
        val_results_df = evaluate(model, val_loader, device)
        
        score_cols = [col for col in val_results_df.columns if 'MAE' not in col and isinstance(val_results_df[col].iloc[0], (int, float))]
        avg_val_score = 0
        if not val_results_df.empty and score_cols:
            avg_val_score = val_results_df[score_cols].mean().mean()

        print("\n--- Epoch {} Validation Report ---".format(epoch + 1))
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS} - 验证集评估结果")
        logger.info(f"{'='*70}")
        if not val_results_df.empty:
            print(val_results_df.to_string(index=False))
            # 记录详细的验证结果
            for _, row in val_results_df.iterrows():
                task_info = f"任务: {row.get('Task', 'N/A')}"
                metrics = []
                for col in val_results_df.columns:
                    if col != 'Task' and isinstance(row[col], (int, float)):
                        metrics.append(f"{col}={row[col]:.4f}")
                logger.info(f"{task_info} | {' | '.join(metrics)}")
        print(f"--- Average Val Score (Higher is better): {avg_val_score:.4f} ---")
        logger.info(f"平均验证分数 (越高越好): {avg_val_score:.6f}")
        logger.info(f"当前学习率: {scheduler.get_last_lr()[0]:.8f}")
        logger.info("-"*70)

        if avg_val_score > best_val_score:
            best_val_score = avg_val_score
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"-> New best model saved! Score improved to: {best_val_score:.4f}\n")
            logger.info(f"🎉 新的最佳模型已保存! 验证分数提升至: {best_val_score:.6f}")
            logger.info(f"模型保存路径: {MODEL_SAVE_PATH}")
        else:
            logger.info(f"验证分数未提升 (当前: {avg_val_score:.6f}, 最佳: {best_val_score:.6f})")
        
        # Update scheduler
        scheduler.step()

    print(f"\n--- Training Finished ---\nBest model saved at: {MODEL_SAVE_PATH}")
    logger.info("\n" + "="*70)
    logger.info("训练完成!")
    logger.info("="*70)
    logger.info(f"最佳验证分数: {best_val_score:.6f}")
    logger.info(f"最佳模型保存路径: {MODEL_SAVE_PATH}")
    logger.info(f"日志文件保存路径: {log_file}")
    logger.info("="*70)

if __name__ == '__main__':
    main()
