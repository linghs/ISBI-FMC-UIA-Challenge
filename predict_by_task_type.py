#!/usr/bin/env python3
"""
按任务类型预测脚本
支持加载按任务类型训练的模型进行多任务预测
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import json
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional, Dict, List
import argparse
from collections import defaultdict

# Import local modules
from model_factory import MultiTaskModelFactory, TASK_CONFIGURATIONS


class InferenceDataset(Dataset):
    """Inference dataset class"""
    
    def __init__(self, data_root: str, transforms: Optional[A.Compose] = None):
        super().__init__()
        self.data_root = data_root
        self.transforms = transforms
        self.csv_path = os.path.join(self.data_root, 'csv_files')
        
        if not os.path.isdir(self.csv_path):
            raise FileNotFoundError(f"CSV path not found: {self.csv_path}")
            
        all_csv_files = glob.glob(os.path.join(self.csv_path, '*.csv'))
        if not all_csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.csv_path}")
            
        df_list = [pd.read_csv(csv_file) for csv_file in all_csv_files]
        self.dataframe = pd.concat(df_list, ignore_index=True).reset_index(drop=True)
        print(f"数据加载完成。总样本数: {len(self.dataframe)}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        record = self.dataframe.iloc[idx]
        task_id = record['task_id']
        task_name = record['task_name']
        
        # Load image
        image_rel_path = record['image_path']
        image_abs_path = os.path.normpath(os.path.join(self.csv_path, image_rel_path))
        image = cv2.imread(image_abs_path)
        
        if image is None:
            print(f"Warning: 无法加载图像 {image_abs_path}")
            # Return next sample
            return self.__getitem__((idx + 1) % len(self))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        
        # Get mask_path (if segmentation task)
        mask_path = None
        if task_name == 'segmentation' and 'mask_path' in record and pd.notna(record['mask_path']):
            mask_path = record['mask_path']
        
        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        
        # Return data including metadata
        return {
            'image': image,
            'task_id': task_id,
            'task_name': task_name,
            'image_path': image_rel_path,
            'mask_path': mask_path,
            'original_size': (original_height, original_width),
            'index': idx
        }


def inference_collate_fn(batch):
    """Inference collate function that preserves metadata"""
    images = torch.stack([item['image'] for item in batch], 0)
    task_ids = [item['task_id'] for item in batch]
    task_names = [item['task_name'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    mask_paths = [item['mask_path'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]
    indices = [item['index'] for item in batch]
    
    return {
        'image': images,
        'task_id': task_ids,
        'task_name': task_names,
        'image_path': image_paths,
        'mask_path': mask_paths,
        'original_size': original_sizes,
        'index': indices
    }


class MultiTaskTypePredictor:
    """
    多任务类型预测器
    支持加载多个按任务类型训练的模型进行预测
    """
    
    def __init__(self, model_configs: List[Dict], encoder_name: str = 'mit_b5', device: str = None):
        """
        初始化预测器
        
        Args:
            model_configs: 模型配置列表，每个配置包含:
                - task_type: 任务类型 (segmentation/classification/Regression/detection)
                - model_path: 模型权重路径
            encoder_name: 编码器名称
            device: 计算设备
        """
        print("="*70)
        print("初始化多任务类型预测器...")
        print("="*70)
        
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"使用设备: {self.device}")
        
        self.encoder_name = encoder_name
        self.model_configs = model_configs
        self.models = {}
        self.task_type_to_task_ids = defaultdict(list)
        
        # 构建任务类型到任务ID的映射
        for cfg in TASK_CONFIGURATIONS:
            self.task_type_to_task_ids[cfg['task_name']].append(cfg['task_id'])
        
        # 加载所有模型
        for config in model_configs:
            task_type = config['task_type']
            model_path = config['model_path']
            
            print(f"\n加载 {task_type} 任务模型...")
            print(f"  模型路径: {model_path}")
            
            if not os.path.exists(model_path):
                print(f"  警告: 模型文件不存在，跳过该模型")
                continue
            
            # 创建模型
            model = MultiTaskModelFactory(
                encoder_name=encoder_name,
                encoder_weights=None,
                task_configs=TASK_CONFIGURATIONS
            ).to(self.device)
            
            # 加载权重
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint)
            model.eval()
            
            self.models[task_type] = model
            
            task_ids = self.task_type_to_task_ids[task_type]
            print(f"  模型加载成功! 包含 {len(task_ids)} 个任务:")
            for task_id in task_ids:
                print(f"    - {task_id}")
        
        # 定义数据预处理
        self.transforms = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        
        # 构建task_id到task_name的映射
        self.task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in TASK_CONFIGURATIONS}
        
        print(f"\n总共加载了 {len(self.models)} 个模型")
        print("="*70)
    
    def predict(self, data_root: str, output_dir: str, batch_size: int = 8):
        """
        对输入数据进行预测
        
        Args:
            data_root: 数据根目录
            output_dir: 输出结果根目录
            batch_size: 批次大小
        """
        print(f"\n{'='*70}")
        print(f"开始预测...")
        print(f"数据目录: {data_root}")
        print(f"输出目录: {output_dir}")
        print(f"批次大小: {batch_size}")
        print(f"{'='*70}\n")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据集
        print(f"加载数据集...")
        dataset = InferenceDataset(data_root=data_root, transforms=self.transforms)
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=inference_collate_fn
        )
        
        # 按任务类型分组数据
        print(f"\n分析数据集任务分布...")
        task_type_samples = defaultdict(int)
        for _, row in dataset.dataframe.iterrows():
            task_name = row['task_name']
            task_type_samples[task_name] += 1
        
        print("任务类型分布:")
        for task_type, count in sorted(task_type_samples.items()):
            print(f"  - {task_type:<15}: {count:>5} 样本")
        
        # 批量推理
        print(f"\n开始推理...")
        classification_results = []
        detection_results = []
        regression_results = []
        task_counts = defaultdict(int)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="预测进度"):
                images = batch['image'].to(self.device)
                task_ids = batch['task_id']
                task_names = batch['task_name']
                image_paths = batch['image_path']
                mask_paths = batch['mask_path']
                original_sizes = batch['original_size']
                
                # 按任务类型分组处理
                task_type_groups = defaultdict(list)
                for i, (task_id, task_name) in enumerate(zip(task_ids, task_names)):
                    task_type_groups[task_name].append({
                        'index': i,
                        'task_id': task_id,
                        'task_name': task_name,
                        'image_path': image_paths[i],
                        'mask_path': mask_paths[i],
                        'original_size': original_sizes[i]
                    })
                
                # 对每种任务类型使用对应的模型
                for task_type, samples in task_type_groups.items():
                    if task_type not in self.models:
                        print(f"\n警告: 没有加载 {task_type} 类型的模型，跳过这些样本")
                        continue
                    
                    model = self.models[task_type]
                    
                    # 按task_id分组（同一任务类型可能有多个task_id）
                    task_id_groups = defaultdict(list)
                    for sample in samples:
                        task_id_groups[sample['task_id']].append(sample)
                    
                    # 对每个task_id进行预测
                    for task_id, task_samples in task_id_groups.items():
                        # 获取该任务的所有图像
                        task_indices = [s['index'] for s in task_samples]
                        task_images = images[task_indices]
                        
                        # 模型推理
                        # 检测任务需要特殊处理（Faster R-CNN）
                        if task_type == 'detection':
                            # 直接调用检测头
                            detection_head = model.heads[task_id]
                            image_list = [task_images[i] for i in range(task_images.shape[0])]
                            
                            # Faster R-CNN 返回预测列表
                            predictions = detection_head(image_list)
                            
                            # 处理每个样本的预测结果
                            for i, sample in enumerate(task_samples):
                                pred = predictions[i]  # Dict with 'boxes', 'labels', 'scores'
                                task_name = sample['task_name']
                                image_path = sample['image_path']
                                original_size = sample['original_size']
                                
                                # 统计
                                task_counts[task_id] += 1
                                
                                # 处理检测结果
                                result = self._process_detection(pred, task_id, image_path, original_size)
                                detection_results.append(result)
                        else:
                            # 其他任务类型正常处理
                            outputs = model(task_images, task_id=task_id)
                            
                            # 处理每个样本的预测结果
                            for i, sample in enumerate(task_samples):
                                pred = outputs[i]
                                task_name = sample['task_name']
                                image_path = sample['image_path']
                                mask_path = sample['mask_path']
                                original_size = sample['original_size']
                                
                                # 统计
                                task_counts[task_id] += 1
                                
                                # 根据任务类型处理结果
                                if task_name == 'segmentation':
                                    self._save_segmentation(pred, image_path, mask_path, output_dir, original_size)
                                
                                elif task_name == 'classification':
                                    result = self._process_classification(pred, task_id, image_path)
                                    classification_results.append(result)
                                
                                elif task_name == 'Regression':
                                    result = self._process_regression(pred, task_id, image_path, original_size)
                                    regression_results.append(result)
        
        # 保存聚合的JSON结果
        print("\n保存预测结果...")
        
        if classification_results:
            json_path = os.path.join(output_dir, 'classification_predictions.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(classification_results, f, indent=2, ensure_ascii=False)
            print(f"  - 分类结果: {json_path} ({len(classification_results)} 样本)")
        
        if detection_results:
            json_path = os.path.join(output_dir, 'detection_predictions.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(detection_results, f, indent=2, ensure_ascii=False)
            print(f"  - 检测结果: {json_path} ({len(detection_results)} 样本)")
        
        if regression_results:
            json_path = os.path.join(output_dir, 'regression_predictions.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(regression_results, f, indent=2, ensure_ascii=False)
            print(f"  - 回归结果: {json_path} ({len(regression_results)} 样本)")
        
        # 打印统计信息
        print(f"\n{'='*70}")
        print("预测完成!")
        print(f"{'='*70}")
        print("\n各任务预测数量:")
        for task_id in sorted(task_counts.keys()):
            task_name_str = self.task_id_to_name.get(task_id, 'unknown')
            count = task_counts[task_id]
            print(f"  - {task_id:<30} ({task_name_str:<15}): {count:>5} 样本")
        print(f"\n总计: {sum(task_counts.values())} 样本")
        print("="*70)
    
    def _save_segmentation(self, pred, image_path, mask_path, output_dir, original_size):
        """保存分割预测结果为图像文件"""
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        
        # 对于多类分割 (C, H, W)，取argmax
        if pred.ndim == 3:
            mask = np.argmax(pred, axis=0).astype(np.uint8)
        else:
            mask = pred.astype(np.uint8)
        
        # 调整回原始尺寸
        h, w = original_size
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 确定输出路径
        if mask_path:
            mask_path_clean = mask_path.replace('../', '')
            output_path = os.path.join(output_dir, mask_path_clean)
        else:
            default_mask_path = image_path.replace('img', 'mask').replace('IMG', 'MASK')
            output_path = os.path.join(output_dir, default_mask_path)
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存mask图像
        cv2.imwrite(output_path, mask)
    
    def _process_classification(self, pred, task_id, image_path):
        """处理分类任务预测结果"""
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        
        # 获取预测类别
        pred_class = int(np.argmax(pred))
        
        # 计算概率分布 (使用softmax)
        pred_exp = np.exp(pred - np.max(pred))
        pred_probs = pred_exp / np.sum(pred_exp)
        
        return {
            'image_path': image_path,
            'task_id': task_id,
            'predicted_class': pred_class,
            'predicted_probs': pred_probs.tolist()
        }
    
    def _process_regression(self, pred, task_id, image_path, original_size):
        """处理回归任务预测结果（关键点定位）"""
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        
        # 归一化坐标
        coords = pred.flatten().tolist()
        
        # 转换为像素坐标
        h, w = original_size
        pixel_coords = []
        for i in range(0, len(coords), 2):
            x_norm, y_norm = coords[i], coords[i+1]
            x_pixel = x_norm * w
            y_pixel = y_norm * h
            pixel_coords.extend([x_pixel, y_pixel])
        
        return {
            'image_path': image_path,
            'task_id': task_id,
            'predicted_points_normalized': coords,
            'predicted_points_pixels': pixel_coords
        }
    
    def _process_detection(self, pred, task_id, image_path, original_size):
        """
        处理检测任务预测结果（Faster R-CNN 输出）
        
        Args:
            pred: Dict with keys 'boxes', 'labels', 'scores' from Faster R-CNN
            task_id: 任务ID
            image_path: 图像路径
            original_size: 原始图像尺寸 (H, W)
        
        Returns:
            Dict with detection results
        """
        # pred is a dict: {'boxes': Tensor[N, 4], 'labels': Tensor[N], 'scores': Tensor[N]}
        boxes = pred['boxes']
        labels = pred['labels']
        scores = pred['scores']
        
        # Convert to numpy
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
            labels = labels.cpu().numpy()
            scores = scores.cpu().numpy()
        
        img_h, img_w = original_size
        
        # 如果有检测结果，选择得分最高的
        if len(boxes) > 0:
            best_idx = np.argmax(scores)
            best_box = boxes[best_idx]  # [x1, y1, x2, y2] in absolute coords (256x256)
            best_score = float(scores[best_idx])
            best_label = int(labels[best_idx])
            
            # 归一化坐标（相对于256x256）
            bbox_norm = [
                float(best_box[0] / 512.0),
                float(best_box[1] / 512.0),
                float(best_box[2] / 512.0),
                float(best_box[3] / 512.0)
            ]
            
            # 转换为原始图像尺寸的像素坐标
            bbox_pixel = [
                float(best_box[0] / 512.0 * img_w),
                float(best_box[1] / 512.0 * img_h),
                float(best_box[2] / 512.0 * img_w),
                float(best_box[3] / 512.0 * img_h)
            ]
            
            # 收集所有检测结果（可选）
            all_detections = []
            for i in range(len(boxes)):
                box = boxes[i]
                all_detections.append({
                    'box_normalized': [
                        float(box[0] / 512.0),
                        float(box[1] / 512.0),
                        float(box[2] / 512.0),
                        float(box[3] / 512.0)
                    ],
                    'box_pixels': [
                        float(box[0] / 512.0 * img_w),
                        float(box[1] / 512.0 * img_h),
                        float(box[2] / 512.0 * img_w),
                        float(box[3] / 512.0 * img_h)
                    ],
                    'label': int(labels[i]),
                    'score': float(scores[i])
                })
        else:
            # 没有检测结果
            bbox_norm = [0.0, 0.0, 0.0, 0.0]
            bbox_pixel = [0.0, 0.0, 0.0, 0.0]
            best_score = 0.0
            best_label = 0
            all_detections = []
        
        return {
            'image_path': image_path,
            'task_id': task_id,
            'bbox_normalized': bbox_norm,
            'bbox_pixels': bbox_pixel,
            'confidence': best_score,
            'label': best_label,
            'num_detections': len(boxes),
            'all_detections': all_detections
        }


def main():
    parser = argparse.ArgumentParser(description='按任务类型多任务预测脚本')
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据根目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--model_dir', type=str, default='task_type_models',
                        help='模型目录（包含按任务类型训练的模型）')
    parser.add_argument('--task_types', type=str, nargs='+',
                        default=['segmentation', 'classification', 'Regression', 'detection'],
                        help='要使用的任务类型列表')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--encoder', type=str, default='mit_b5',
                        help='编码器名称')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备 (cuda/cpu)，默认自动选择')
    
    args = parser.parse_args()
    
    # 构建模型配置列表
    model_configs = []
    for task_type in args.task_types:
        model_path = os.path.join(args.model_dir, f'{task_type}_best_model.pth')
        model_configs.append({
            'task_type': task_type,
            'model_path': model_path
        })
    
    # 创建预测器
    predictor = MultiTaskTypePredictor(
        model_configs=model_configs,
        encoder_name=args.encoder,
        device=args.device
    )
    
    # 执行预测
    predictor.predict(
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    
    print("\n预测完成!")


if __name__ == '__main__':
    main()

