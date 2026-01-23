"""
Test script for classification/regression models with task-specific metrics.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
import argparse
import yaml
import importlib
import json
from pathlib import Path
from models.losses import DiceLoss, CombinedLoss, FocalLoss

def parse_args():
    parser = argparse.ArgumentParser(description='Test model with task-specific metrics')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--checkpoint', type=str, default='best.pth', help='Checkpoint to load (best.pth or last.pth)')
    parser.add_argument('--save_predictions', action='store_true', help='Save predictions to file')
    
    return parser.parse_args()


def resolve_model_class(dotted_path: str):
    """Dynamically import model class from dotted path."""
    module_name, class_name = dotted_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    if hasattr(module, class_name):
        return getattr(module, class_name)
    
    candidates = [class_name.upper(), class_name.lower(), class_name.title()]
    for cand in candidates:
        if hasattr(module, cand):
            return getattr(module, cand)
    raise AttributeError(f"Class '{class_name}' not found in module '{module_name}'.")


# ========== METRICS ==========

def compute_iou_boxes(pred_boxes, target_boxes):
    """
    Compute IoU (Intersection over Union) for bounding boxes.
    
    Args:
        pred_boxes: Tensor of shape (N, 4) with format [x_min, y_min, x_max, y_max]
        target_boxes: Tensor of shape (N, 4) with format [x_min, y_min, x_max, y_max]
    
    Returns:
        iou: Tensor of shape (N,) with IoU for each box
    """
    # Intersection coordinates
    x1_inter = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1_inter = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2_inter = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2_inter = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    
    # Intersection area
    inter_width = torch.clamp(x2_inter - x1_inter, min=0)
    inter_height = torch.clamp(y2_inter - y1_inter, min=0)
    inter_area = inter_width * inter_height
    
    # Union area
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union_area = pred_area + target_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-6)
    
    return iou


def compute_dice_boxes(pred_boxes, target_boxes):
    """
    Compute Dice coefficient for bounding boxes.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Args:
        pred_boxes: Tensor of shape (N, 4) with format [x_min, y_min, x_max, y_max]
        target_boxes: Tensor of shape (N, 4) with format [x_min, y_min, x_max, y_max]
    
    Returns:
        dice: Tensor of shape (N,) with Dice coefficient for each box
    """
    # Intersection coordinates
    x1_inter = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1_inter = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2_inter = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2_inter = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    
    # Intersection area
    inter_width = torch.clamp(x2_inter - x1_inter, min=0)
    inter_height = torch.clamp(y2_inter - y1_inter, min=0)
    inter_area = inter_width * inter_height
    
    # Areas
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    
    # Dice
    dice = (2 * inter_area) / (pred_area + target_area + 1e-6)
    
    return dice


def compute_box_center_distance(pred_boxes, target_boxes):
    """
    Compute Euclidean distance between box centers (normalized by image size).
    
    Args:
        pred_boxes: Tensor of shape (N, 4)
        target_boxes: Tensor of shape (N, 4)
    
    Returns:
        distances: Tensor of shape (N,)
    """
    pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    
    target_center_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_center_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    
    distances = torch.sqrt((pred_center_x - target_center_x)**2 + (pred_center_y - target_center_y)**2)
    
    return distances


def compute_box_size_error(pred_boxes, target_boxes):
    """
    Compute relative size error (width and height).
    
    Returns:
        width_error: Mean relative width error
        height_error: Mean relative height error
    """
    pred_width = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_height = pred_boxes[:, 3] - pred_boxes[:, 1]
    
    target_width = target_boxes[:, 2] - target_boxes[:, 0]
    target_height = target_boxes[:, 3] - target_boxes[:, 1]
    
    width_error = torch.abs(pred_width - target_width) / (target_width + 1e-6)
    height_error = torch.abs(pred_height - target_height) / (target_height + 1e-6)
    
    return width_error, height_error


class MetricsComputer:
    """Compute task-specific metrics."""
    
    def __init__(self, task='box'):
        self.task = task
        self.metrics = {}
    
    def compute(self, predictions, targets):
        """Compute metrics based on task type."""
        if self.task == 'box':
            return self._compute_box_metrics(predictions, targets)
        elif self.task == 'classification':
            return self._compute_classification_metrics(predictions, targets)
        elif self.task in ['segmentation', 'contours']:
            return self._compute_segmentation_metrics(predictions, targets)
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def _compute_box_metrics(self, pred_boxes, target_boxes):
        """Compute bounding box metrics."""
        # Ensure boxes are in [x_min, y_min, x_max, y_max] format
        # If your boxes are normalized [0,1], keep them as is
        # If they're in pixels, you might want to normalize
        
        iou = compute_iou_boxes(pred_boxes, target_boxes)
        dice = compute_dice_boxes(pred_boxes, target_boxes)
        center_dist = compute_box_center_distance(pred_boxes, target_boxes)
        width_err, height_err = compute_box_size_error(pred_boxes, target_boxes)
        
        return {
            'iou': iou.mean().item(),
            'dice': dice.mean().item(),
            'center_distance': center_dist.mean().item(),
            'width_error': width_err.mean().item(),
            'height_error': height_err.mean().item(),
            'iou_std': iou.std().item(),
            'dice_std': dice.std().item(),
        }
    
    def _compute_classification_metrics(self, predictions, targets):
        """Compute classification metrics."""
        preds = predictions.argmax(dim=1)
        correct = (preds == targets).float()
        
        return {
            'accuracy': correct.mean().item(),
            'top1_error': (1 - correct.mean()).item(),
        }
    
    def _compute_segmentation_metrics(self, predictions, targets):
        """Compute segmentation metrics (IoU, Dice)."""
        # For segmentation, predictions are logits [B, C, H, W]
        preds = predictions.argmax(dim=1)  # [B, H, W]
        
        # Compute per-class IoU
        num_classes = predictions.shape[1]
        ious = []
        dices = []
        
        for cls in range(num_classes):
            pred_mask = (preds == cls)
            target_mask = (targets == cls)
            
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            
            iou = (intersection / (union + 1e-6)).item()
            dice = (2 * intersection / (pred_mask.sum() + target_mask.sum() + 1e-6)).item()
            
            ious.append(iou)
            dices.append(dice)
        
        return {
            'mean_iou': np.mean(ious),
            'mean_dice': np.mean(dices),
            'iou_per_class': ious,
            'dice_per_class': dices,
        }


def test_model(model, loader, criterion, metrics_computer, device, save_predictions=False):
    """Test model and compute metrics."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    running_loss = 0
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Testing"):
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            
            # Store for metrics computation
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    metrics = metrics_computer.compute(all_predictions, all_targets)
    metrics['test_loss'] = running_loss / len(loader)
    
    # Optionally save predictions
    if save_predictions:
        predictions_data = {
            'predictions': all_predictions.numpy(),
            'targets': all_targets.numpy(),
        }
        return metrics, predictions_data
    
    return metrics, None


def save_metrics_to_file(metrics, save_path, format='json'):
    """Save metrics to file (JSON or CSV)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if format == 'json':
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"📊 Metrics saved to: {save_path}")
    
    elif format == 'csv':
        import csv
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for key, value in metrics.items():
                if isinstance(value, (list, np.ndarray)):
                    # For per-class metrics
                    for i, v in enumerate(value):
                        writer.writerow([f"{key}_class_{i}", v])
                else:
                    writer.writerow([key, value])
        print(f"📊 Metrics saved to: {save_path}")


def print_metrics(metrics, model_name):
    """Pretty print metrics."""
    print("\n" + "="*80)
    print(f"TEST RESULTS: {model_name}")
    print("="*80)
    
    for key, value in metrics.items():
        if isinstance(value, (list, np.ndarray)):
            print(f"{key}:")
            for i, v in enumerate(value):
                print(f"  Class {i}: {v:.6f}")
        else:
            print(f"{key}: {value:.6f}")
    
    print("="*80 + "\n")


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Normalize config
    if 'model_args' in config and isinstance(config['model_args'], list):
        merged = {}
        for item in config['model_args']:
            if isinstance(item, dict):
                merged.update(item)
        config['model_args'] = merged
    
    if 'data_args' in config and isinstance(config['data_args'], list):
        merged = {}
        for item in config['data_args']:
            if isinstance(item, dict):
                merged.update(item)
        config['data_args'] = merged
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on: {device}")
    
    # Load model
    ModelClass = resolve_model_class(config['model'])
    model_kwargs = config.get('model_args', {})
    model = ModelClass(**model_kwargs).to(device)
    
    # Load checkpoint
    model_path = config.get('path', 'models/')
    checkpoint_path = os.path.join(model_path, args.checkpoint)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Load data
    from dataloader_segmentation import get_oxford_loaders
    
    data_path = config.get('data_path', 'data/oxford-iiit-pet')
    task = config.get('task', 'box')
    batch_size = config.get('batch_size', 32)
    
    data_kwargs = {'task': task, 'batch_size': batch_size}
    if config.get('data_args'):
        data_kwargs.update(config['data_args'])
    
    loaders = get_oxford_loaders(data_path, **data_kwargs)
    
    n_classes = model_kwargs.get('n_classes', 3)

    # Setup criterion
    criterion_map = {
        'CrossEntropyLoss': nn.CrossEntropyLoss(),
        'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(),
        'MSELoss': nn.MSELoss(),
        'DiceLoss': DiceLoss(n_classes=n_classes),
        'CombinedLoss': CombinedLoss(n_classes=n_classes, ce_weight=0.5, dice_weight=0.5),
        'FocalLoss': FocalLoss(alpha=1.0, gamma=2.0),
    }
    criterion_name = config.get('criterion', 'MSELoss')
    criterion = criterion_map[criterion_name]
    
    # Setup metrics computer
    metrics_computer = MetricsComputer(task=task)
    
    # Test model
    print(f"\n🧪 Testing model on task: {task}")
    metrics, predictions = test_model(
        model, loaders['test'], criterion, metrics_computer, 
        device, save_predictions=args.save_predictions
    )
    
    # Get model name
    model_name = Path(model_path).name
    
    # Print metrics
    print_metrics(metrics, model_name)
    
    # Save metrics
    metrics_save_path = os.path.join(model_path, 'test_metrics.json')
    save_metrics_to_file(metrics, metrics_save_path, format='json')
    
    # Also save as CSV for easy reading
    csv_save_path = os.path.join(model_path, 'test_metrics.csv')
    save_metrics_to_file(metrics, csv_save_path, format='csv')
    
    # Save predictions if requested
    if args.save_predictions and predictions:
        pred_save_path = os.path.join(model_path, 'predictions.npz')
        np.savez(pred_save_path, **predictions)
        print(f"💾 Predictions saved to: {pred_save_path}")
    
    print("✅ Testing complete!")


if __name__ == '__main__':
    main()
