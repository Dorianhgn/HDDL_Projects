"""
Compare and visualize results from different classification models.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import yaml


def load_loss_log(path):
    """Load loss log from a training directory."""
    loss_file = Path(path) / 'loss_log.npy'
    if loss_file.exists():
        return np.load(loss_file, allow_pickle=True).item()
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
    
    # Load results from each model
    for config_path in config_files:
        if not os.path.exists(config_path):
            continue
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_dir = config.get('path', '')
        model_name = get_model_name(config_path)
        
        data = load_loss_log(model_dir)
        results[model_name] = data
        
        if data:
            print(f"✓ Loaded results for: {model_name}")
        else:
            print(f"✗ No results found for: {model_name} (not trained yet)")
    
    if not any(results.values()):
        print("\n⚠ No trained models found. Train models first using:")
        print("  python training_classification.py --config <config_file>")
        print("  or")
        print("  ./train_all_classification.sh")
        return
    
    # Print summary
    print_summary_table(results)
    
    # Analyze convergence
    analyze_convergence(results)
    
    # Plot comparison
    plot_comparison(results)
    
    # Find best model
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
        print(f"🏆 BEST MODEL: {best_model}")
        print(f"   Best validation loss: {best_loss:.6f}")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
