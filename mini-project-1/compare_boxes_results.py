"""
Compare and visualize results from different classification models.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import yaml
import json


def load_loss_log(path):
    """Load loss log from a training directory."""
    loss_file = Path(path) / 'loss_log.npy'
    if loss_file.exists():
        return np.load(loss_file, allow_pickle=True).item()
    return None


def load_test_metrics(path):
    """Load test metrics from a training directory."""
    metrics_file = Path(path) / 'test_metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None


def get_model_name(config_path):
    """Extract model name from config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_path = config.get('path', '')
    model_name = Path(model_path).name
    return model_name if model_name else config_path.stem


def plot_comparison(results):
    """Plot training curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    ax = axes[0]
    for name, data in results.items():
        if data and 'train' in data:
            ax.plot(data['train'], label=name, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot validation loss
    ax = axes[1]
    for name, data in results.items():
        if data and 'val' in data:
            ax.plot(data['val'], label=name, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/temp_results/boxes/comparison.png', dpi=150)
    print(f"\nComparison plot saved to: models/temp_results/boxes/comparison.png")
    plt.show()


def print_summary_table(results):
    """Print a summary table of all results."""
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<35} {'Best Train':<12} {'Best Val':<12} {'Final Train':<12} {'Final Val':<12}")
    print("-"*80)
    
    for name, data in results.items():
        if data:
            train_loss = data.get('train', [])
            val_loss = data.get('val', [])
            
            best_train = min(train_loss) if train_loss else float('nan')
            best_val = min(val_loss) if val_loss else float('nan')
            final_train = train_loss[-1] if train_loss else float('nan')
            final_val = val_loss[-1] if val_loss else float('nan')
            
            print(f"{name:<35} {best_train:<12.6f} {best_val:<12.6f} {final_train:<12.6f} {final_val:<12.6f}")
        else:
            print(f"{name:<35} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
    
    print("="*80)


def analyze_convergence(results):
    """Analyze convergence speed and final performance."""
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS")
    print("="*80)
    
    for name, data in results.items():
        if data and 'val' in data:
            val_loss = np.array(data['val'])
            
            # Find epoch of best validation loss
            best_epoch = np.argmin(val_loss)
            best_loss = val_loss[best_epoch]
            
            # Calculate improvement from first to best
            if len(val_loss) > 0:
                improvement = ((val_loss[0] - best_loss) / val_loss[0]) * 100
                
                print(f"\n{name}:")
                print(f"  Best epoch: {best_epoch + 1}/{len(val_loss)}")
                print(f"  Best val loss: {best_loss:.6f}")
                print(f"  Improvement: {improvement:.2f}%")
                
                # Check for overfitting
                if len(data.get('train', [])) > 0:
                    train_loss = data['train']
                    gap = train_loss[-1] - val_loss[-1]
                    if gap < -0.01:  # Val loss significantly lower than train
                        print(f"  ⚠ Possible underfitting (gap: {gap:.4f})")
                    elif gap > 0.05:  # Train loss significantly lower than val
                        print(f"  ⚠ Possible overfitting (gap: {gap:.4f})")
                    else:
                        print(f"  ✓ Good train/val balance (gap: {gap:.4f})")


def print_test_metrics_table(test_metrics):
    """Print a summary table of test metrics for all models."""
    if not test_metrics or not any(test_metrics.values()):
        print("\n⚠️  No test metrics available. Run test_classification.py first.")
        return
    
    print("\n" + "="*100)
    print("MODEL EVALUATION COMPARISON SUMMARY")
    print("="*100)
    
    # Determine available metrics from first model with data
    available_metrics = set()
    for metrics in test_metrics.values():
        if metrics:
            available_metrics.update(metrics.keys())
    
    # Filter out list-type metrics for summary table
    scalar_metrics = {m for m in available_metrics 
                      if not any(m.endswith(suffix) for suffix in ['_per_class', '_std'])}
    
    # Sort metrics for consistent display
    metric_order = ['test_loss', 'iou', 'dice', 'accuracy', 'center_distance', 
                    'width_error', 'height_error', 'mean_iou', 'mean_dice']
    sorted_metrics = [m for m in metric_order if m in scalar_metrics]
    sorted_metrics += [m for m in sorted(scalar_metrics) if m not in sorted_metrics]
    
    # Print header
    header = f"{'Model':<35}"
    for metric in sorted_metrics:
        header += f" {metric.replace('_', ' ').title():<15}"
    print(header)
    print("-"*100)
    
    # Print data
    for name, metrics in test_metrics.items():
        if metrics:
            row = f"{name:<35}"
            for metric in sorted_metrics:
                value = metrics.get(metric, float('nan'))
                if isinstance(value, (int, float)):
                    row += f" {value:<15.6f}"
                else:
                    row += f" {'N/A':<15}"
            print(row)
        else:
            row = f"{name:<35}"
            for _ in sorted_metrics:
                row += f" {'Not tested':<15}"
            print(row)
    
    print("="*100)
    
    # Print additional details (std dev, per-class metrics if available)
    print("\nADDITIONAL METRICS:")
    print("-"*100)
    for name, metrics in test_metrics.items():
        if metrics:
            has_additional = False
            additional_info = []
            
            # Check for std metrics
            for key in ['iou_std', 'dice_std']:
                if key in metrics:
                    has_additional = True
                    additional_info.append(f"{key}: {metrics[key]:.6f}")
            
            # Check for per-class metrics
            for key in ['iou_per_class', 'dice_per_class']:
                if key in metrics and isinstance(metrics[key], list):
                    has_additional = True
                    values_str = ", ".join([f"{v:.4f}" for v in metrics[key]])
                    additional_info.append(f"{key}: [{values_str}]")
            
            if has_additional:
                print(f"\n{name}:")
                for info in additional_info:
                    print(f"  {info}")
    
    if not any(test_metrics.values()):
        print("  No additional metrics available.")
    print()


def plot_test_metrics_comparison(test_metrics, key_metrics=['iou', 'dice', 'accuracy']):
    """Plot bar chart comparing key test metrics across models."""
    models = []
    metrics_data = {metric: [] for metric in key_metrics}
    
    for name, metrics in test_metrics.items():
        if metrics:
            models.append(name)
            for metric in key_metrics:
                value = metrics.get(metric, 0)
                metrics_data[metric].append(value if value else 0)
    
    if not models:
        print("No test metrics available for plotting.")
        return
    
    # Filter out metrics with all zeros
    active_metrics = {k: v for k, v in metrics_data.items() if any(v)}
    
    if not active_metrics:
        print("No non-zero metrics to plot.")
        return
    
    # Create bar chart
    x = np.arange(len(models))
    width = 0.8 / len(active_metrics)
    
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.5), 6))
    
    for i, (metric, values) in enumerate(active_metrics.items()):
        offset = width * i - width * (len(active_metrics) - 1) / 2
        ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Test Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('models/temp_results/boxes/test_metrics_comparison.png', dpi=150)
    print(f"\nTest metrics comparison plot saved to: models/temp_results/boxes/test_metrics_comparison.png")
    plt.show()



def main():
    # Configuration files to analyze
    config_files = [
        "models/config_unet_encoder.yaml",
        "models/config_vgg_head_only.yaml",
        "models/config_vgg_head_last_conv.yaml",
        "models/config_vgg_full_finetune.yaml",
        "models/config_resnet18_head_only.yaml",
        "models/config_resnet18_full_finetune.yaml",
        "models/config_resnet50_head_only.yaml",
    ]
    
    results = {}
    test_metrics = {}
    
    # Load results from each model
    for config_path in config_files:
        if not os.path.exists(config_path):
            continue
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_dir = config.get('path', '')
        model_name = get_model_name(config_path)
        
        # Load training results
        data = load_loss_log(model_dir)
        results[model_name] = data
        
        # Load test metrics
        test_data = load_test_metrics(model_dir)
        test_metrics[model_name] = test_data
        
        if data:
            print(f"✓ Loaded training results for: {model_name}")
        else:
            print(f"✗ No training results found for: {model_name} (not trained yet)")
        
        if test_data:
            print(f"  ✓ Test metrics available")
        else:
            print(f"  ✗ No test metrics (run: python test_classification.py --config {config_path})")
    
    if not any(results.values()):
        print("\n⚠ No trained models found. Train models first using:")
        print("  python training_classification.py --config <config_file>")
        print("  or")
        print("  ./train_all_classification.sh")
        return
    
    # ==== TRAINING RESULTS ====
    print("\n" + "="*100)
    print("TRAINING RESULTS")
    print("="*100)
    
    # Print summary
    print_summary_table(results)
    
    # Analyze convergence
    analyze_convergence(results)
    
    # Plot comparison
    plot_comparison(results)
    
    # Find best model by validation loss
    best_model = None
    best_loss = float('inf')
    
    for name, data in results.items():
        if data and 'val' in data:
            min_loss = min(data['val'])
            if min_loss < best_loss:
                best_loss = min_loss
                best_model = name
    
    if best_model:
        print(f"\n{'='*80}")
        print(f"🏆 BEST MODEL (by val loss): {best_model}")
        print(f"   Best validation loss: {best_loss:.6f}")
        print(f"{'='*80}\n")
    
    # ==== TEST RESULTS ====
    if any(test_metrics.values()):
        print("\n" + "="*100)
        print("TEST RESULTS")
        print("="*100)
        
        print_test_metrics_table(test_metrics)
        
        # Plot test metrics
        plot_test_metrics_comparison(test_metrics)
        
        # Find best model by test metric (e.g., IoU for boxes)
        best_test_model = None
        best_test_score = -float('inf')
        best_metric_name = None
        
        # Determine which metric to use (IoU for boxes, accuracy for classification, etc.)
        for metric_name in ['iou', 'dice', 'accuracy', 'mean_iou']:
            for name, metrics in test_metrics.items():
                if metrics and metric_name in metrics:
                    score = metrics[metric_name]
                    if score > best_test_score:
                        best_test_score = score
                        best_test_model = name
                        best_metric_name = metric_name
        
        if best_test_model:
            print(f"\n{'='*100}")
            print(f"🏆 BEST MODEL (by test {best_metric_name}): {best_test_model}")
            print(f"   Best {best_metric_name}: {best_test_score:.6f}")
            print(f"{'='*100}\n")
    else:
        print("\n" + "="*100)
        print("⚠️  No test results available yet.")
        print("="*100)
        print("\nTo test models, run:")
        print("  python test_classification.py --config <config_file>")
        print("\nOr test all models:")
        for config in config_files:
            if os.path.exists(config):
                print(f"  python test_classification.py --config {config}")
        print()


if __name__ == '__main__':
    main()
