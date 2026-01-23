import random
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

from config import Config


# ============================================================================
# Visualisations
# ============================================================================

def plot_training_comparison(history_resnet, history_vit, save_path=None):
    """Compare les courbes d'entraînement ResNet vs ViT"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history_resnet['val_acc']) + 1)
    
    # Accuracy
    axes[0].plot(epochs, history_resnet['val_acc'], 'o-', label='ResNet-50', linewidth=2, markersize=8)
    axes[0].plot(epochs, history_vit['val_acc'], 's-', label='ViT-Base/16', linewidth=2, markersize=8)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[0].set_title('Validation Accuracy: ResNet vs ViT', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(epochs, history_resnet['val_loss'], 'o-', label='ResNet-50', linewidth=2, markersize=8)
    axes[1].plot(epochs, history_vit['val_loss'], 's-', label='ViT-Base/16', linewidth=2, markersize=8)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Loss', fontsize=12)
    axes[1].set_title('📉 Validation Loss: ResNet vs ViT', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée: {save_path}")
    
    plt.show()


def visualize_puzzle_effect(puzzle_transform, val_dataset, num_samples=4, save_path=None):
    """
    Visualise l'effet de la transformation puzzle sur des images réelles.
    
    Affiche côte à côte:
    - Image originale
    - Image avec puzzle appliqué
    """
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4*num_samples))
    
    # Sélectionne des images aléatoires
    indices = random.sample(range(len(val_dataset)), num_samples)
    
    # Transformations pour affichage (sans normalisation)
    to_pil = transforms.ToPILImage()
    resize = transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE))
    
    for i, idx in enumerate(indices):
        # Charge l'image originale (sans transformation du dataset)
        img_path = val_dataset.imgs[idx][0]
        img_original = Image.open(img_path).convert('RGB')
        img_original = resize(img_original)
        
        # Applique le puzzle
        img_tensor = transforms.ToTensor()(img_original)
        img_puzzled = puzzle_transform(img_tensor)
        img_puzzled = to_pil(img_puzzled)
        
        # Affiche
        axes[i, 0].imshow(img_original)
        axes[i, 0].set_title(f'Original - Class: {val_dataset.classes[val_dataset.imgs[idx][1]]}', 
                             fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(img_puzzled)
        axes[i, 1].set_title(f'Puzzled (Seed={Config.SEED_PERMUTATION})', fontsize=10)
        axes[i, 1].axis('off')
    
    plt.suptitle('Effet de la Transformation Puzzle', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée: {save_path}")
    
    plt.show()