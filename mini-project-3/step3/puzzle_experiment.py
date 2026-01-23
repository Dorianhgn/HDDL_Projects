"""
🧩 The Puzzle Challenge: ResNet-50 vs ViT-Base/16
==================================================

Expérience comparative démontrant la différence de biais inductif entre:
- ResNet-50 (CNN): Structure locale → Échoue avec permutation spatiale
- ViT-Base/16 (Transformer): Structure globale → S'adapte à la permutation

Entraîne 4 modèles:
1. ResNet-50 sur données ORIGINALES
2. ViT-Base/16 sur données ORIGINALES  
3. ResNet-50 sur données PUZZLE
4. ViT-Base/16 sur données PUZZLE

Auteur: Équipe HDDL (Lise, Matteo, Sara, Dorian)
Date: Janvier 2026
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Imports des modules locaux
from config import Config
from dataloader import create_dataloaders
from models import create_resnet50, create_vit_base
from train_and_test import train_model
from visualisations import plot_training_comparison, visualize_puzzle_effect


# ============================================================================
# Fonction de Comparaison Multi-Modèles
# ============================================================================

def plot_all_models_comparison(histories, save_path=None):
    """
    Compare les courbes d'entraînement des 4 modèles.
    
    Args:
        histories: Dict avec clés 'ResNet_Original', 'ViT_Original', 
                   'ResNet_Puzzle', 'ViT_Puzzle'
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs = range(1, Config.NUM_EPOCHS + 1)
    
    # Couleurs et styles
    colors = {
        'ResNet_Original': '#e74c3c',
        'ViT_Original': '#3498db',
        'ResNet_Puzzle': '#e67e22',
        'ViT_Puzzle': '#2ecc71'
    }
    
    markers = {
        'ResNet_Original': 'o',
        'ViT_Original': 's',
        'ResNet_Puzzle': '^',
        'ViT_Puzzle': 'D'
    }
    
    # 1. Validation Accuracy - Tous les modèles
    for model_name, history in histories.items():
        axes[0, 0].plot(epochs, history['val_acc'], 
                       marker=markers[model_name], 
                       label=model_name.replace('_', ' '),
                       linewidth=2, markersize=6, 
                       color=colors[model_name])
    
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[0, 0].set_title('Validation Accuracy: Tous les Modèles', 
                         fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Validation Loss - Tous les modèles
    for model_name, history in histories.items():
        axes[0, 1].plot(epochs, history['val_loss'],
                       marker=markers[model_name],
                       label=model_name.replace('_', ' '),
                       linewidth=2, markersize=6,
                       color=colors[model_name])
    
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Validation Loss', fontsize=12)
    axes[0, 1].set_title('📉 Validation Loss: Tous les Modèles',
                         fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Comparaison Original vs Puzzle pour ResNet
    axes[1, 0].plot(epochs, histories['ResNet_Original']['val_acc'],
                   'o-', label='ResNet Original', linewidth=2, markersize=8,
                   color=colors['ResNet_Original'])
    axes[1, 0].plot(epochs, histories['ResNet_Puzzle']['val_acc'],
                   '^-', label='ResNet Puzzle', linewidth=2, markersize=8,
                   color=colors['ResNet_Puzzle'])
    
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[1, 0].set_title('ResNet-50: Original vs Puzzle',
                         fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Comparaison Original vs Puzzle pour ViT
    axes[1, 1].plot(epochs, histories['ViT_Original']['val_acc'],
                   's-', label='ViT Original', linewidth=2, markersize=8,
                   color=colors['ViT_Original'])
    axes[1, 1].plot(epochs, histories['ViT_Puzzle']['val_acc'],
                   'D-', label='ViT Puzzle', linewidth=2, markersize=8,
                   color=colors['ViT_Puzzle'])
    
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[1, 1].set_title('ViT-Base/16: Original vs Puzzle',
                         fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Figure sauvegardée: {save_path}")
    
    plt.show()


# ============================================================================
# Script Principal
# ============================================================================

def main():
    """Fonction principale exécutant l'expérience complète"""
    
    print("\n" + "="*70)
    print("THE PUZZLE CHALLENGE: ResNet-50 vs ViT-Base/16")
    print("="*70)
    print("Entraînement de 4 modèles:")
    print("  1. ResNet-50 sur données ORIGINALES")
    print("  2. ViT-Base/16 sur données ORIGINALES")
    print("  3. ResNet-50 sur données PUZZLE")
    print("  4. ViT-Base/16 sur données PUZZLE")
    print("="*70 + "\n")
    
    # Configuration
    print(f"Configuration:")
    print(f"   - Device: {Config.DEVICE}")
    print(f"   - Image size: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
    print(f"   - Patch size: {Config.PATCH_SIZE}x{Config.PATCH_SIZE}")
    print(f"   - Batch size: {Config.BATCH_SIZE}")
    print(f"   - Epochs: {Config.NUM_EPOCHS}")
    print(f"   - Learning rate: {Config.LEARNING_RATE}")
    
    # Crée les dossiers de résultats
    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Étape 1: Charge les données ORIGINALES (sans puzzle)
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 1: Chargement des données ORIGINALES")
    print("="*70)
    
    train_loader_orig, val_loader_orig, _ = create_dataloaders(use_puzzle=False)
    
    # ========================================================================
    # Étape 2: Charge les données PUZZLE
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 2: Chargement des données PUZZLE")
    print("="*70)
    
    train_loader_puzzle, val_loader_puzzle, puzzle_transform = create_dataloaders(use_puzzle=True)
    
    # ========================================================================
    # Étape 3: Visualise l'effet du puzzle
    # ========================================================================
    print("\n" + "="*70)
    print("Visualisation de la Transformation Puzzle")
    print("="*70)
    
    # Crée un dataset temporaire pour visualisation
    from torchvision.datasets import ImageFolder
    dataset_path = Config.DATA_DIR / "imagenette2"
    val_dataset_viz = ImageFolder(dataset_path / "val", transform=None)
    
    visualize_puzzle_effect(
        puzzle_transform,
        val_dataset_viz,
        num_samples=4,
        save_path=Config.RESULTS_DIR / "puzzle_effect.png"
    )
    
    # ========================================================================
    # Étape 4: Entraîne les 4 modèles
    # ========================================================================
    
    histories = {}
    
    # MODÈLE 1: ResNet-50 sur données ORIGINALES
    print("\n" + "="*70)
    print("MODÈLE 1/4: ResNet-50 sur données ORIGINALES")
    print("="*70)
    resnet_orig = create_resnet50(num_classes=Config.NUM_CLASSES, freeze_backbone=True)
    histories['ResNet_Original'] = train_model(
        resnet_orig,
        "ResNet50_Original",
        train_loader_orig,
        val_loader_orig,
        num_epochs=Config.NUM_EPOCHS
    )
    
    # MODÈLE 2: ViT-Base/16 sur données ORIGINALES
    print("\n" + "="*70)
    print("MODÈLE 2/4: ViT-Base/16 sur données ORIGINALES")
    print("="*70)
    vit_orig = create_vit_base(num_classes=Config.NUM_CLASSES, freeze_backbone=True)
    histories['ViT_Original'] = train_model(
        vit_orig,
        "ViT_Base_Original",
        train_loader_orig,
        val_loader_orig,
        num_epochs=Config.NUM_EPOCHS
    )
    
    # MODÈLE 3: ResNet-50 sur données PUZZLE
    print("\n" + "="*70)
    print("MODÈLE 3/4: ResNet-50 sur données PUZZLE")
    print("="*70)
    resnet_puzzle = create_resnet50(num_classes=Config.NUM_CLASSES, freeze_backbone=True)
    histories['ResNet_Puzzle'] = train_model(
        resnet_puzzle,
        "ResNet50_Puzzle",
        train_loader_puzzle,
        val_loader_puzzle,
        num_epochs=Config.NUM_EPOCHS
    )
    
    # MODÈLE 4: ViT-Base/16 sur données PUZZLE
    print("\n" + "="*70)
    print("MODÈLE 4/4: ViT-Base/16 sur données PUZZLE")
    print("="*70)
    vit_puzzle = create_vit_base(num_classes=Config.NUM_CLASSES, freeze_backbone=True)
    histories['ViT_Puzzle'] = train_model(
        vit_puzzle,
        "ViT_Base_Puzzle",
        train_loader_puzzle,
        val_loader_puzzle,
        num_epochs=Config.NUM_EPOCHS
    )
    
    # ========================================================================
    # Étape 5: Compare tous les résultats
    # ========================================================================
    print("\n" + "="*70)
    print("ANALYSE COMPARATIVE DES 4 MODÈLES")
    print("="*70)
    
    # Visualisation comparative
    plot_all_models_comparison(
        histories,
        save_path=Config.RESULTS_DIR / "all_models_comparison.png"
    )
    
    # Tableau récapitulatif
    print("\n🏆 RÉSULTATS FINAUX (Époque {}):\n".format(Config.NUM_EPOCHS))
    print("┌─────────────────────────┬──────────────┬──────────────┐")
    print("│ Modèle                  │ Val Accuracy │ Val Loss     │")
    print("├─────────────────────────┼──────────────┼──────────────┤")
    
    for model_name in ['ResNet_Original', 'ViT_Original', 'ResNet_Puzzle', 'ViT_Puzzle']:
        acc = histories[model_name]['val_acc'][-1]
        loss = histories[model_name]['val_loss'][-1]
        name_display = model_name.replace('_', ' ')
        print(f"│ {name_display:<23} │ {acc:>11.2f}% │ {loss:>12.4f} │")
    
    print("└─────────────────────────┴──────────────┴──────────────┘\n")
    
    # Sauvegarde tous les historiques
    for model_name, history in histories.items():
        json_path = Config.RESULTS_DIR / f"history_{model_name}.json"
        with open(json_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    print(f"\n💾 Historiques sauvegardés dans {Config.RESULTS_DIR}")


if __name__ == "__main__":
    # Set random seeds pour la reproductibilité
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Lance l'expérience
    main()

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import requests
import tarfile
from PIL import Image


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration centralisée de l'expérience"""
    
    # Chemins
    DATA_DIR = Path("./data")
    RESULTS_DIR = Path("./results")
    
    # Dataset
    DATASET_NAME = "imagenette2"  # Version 320px
    DATASET_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
    NUM_CLASSES = 10
    
    # Image Processing
    IMG_SIZE = 224
    PATCH_SIZE = 16
    GRID_SIZE = IMG_SIZE // PATCH_SIZE  # 14x14 grid
    NUM_PATCHES = GRID_SIZE * GRID_SIZE  # 196 patches
    
    # Transformation Puzzle
    SEED_PERMUTATION = 42  # Seed pour la permutation fixe
    
    # Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ImageNet Stats
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]


# ========================
# 1. Transformation Puzzle
# ========================

class PuzzlePermutation:
    """
    Transformation qui découpe l'image en grille de patches et les permute.
    
    La permutation est FIXE pour tout le dataset (même seed) pour assurer
    que tous les modèles voient exactement la même distorsion.
    """
    
    def __init__(self, img_size=224, patch_size=16, seed=42):
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # 14
        self.num_patches = self.grid_size ** 2  # 196
        
        # Génère la permutation FIXE une seule fois
        rng = random.Random(seed)
        self.permutation = list(range(self.num_patches))
        rng.shuffle(self.permutation)
    
    def __call__(self, img):
        """
        Applique la permutation puzzle à une image PIL ou Tensor.
        
        Args:
            img: PIL Image ou Tensor (C, H, W)
        
        Returns:
            Tensor permuté (C, H, W)
        """
        # Convertit PIL → Tensor si nécessaire
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
        
        C, H, W = img.shape
        assert H == W == self.img_size, f"Image doit être {self.img_size}x{self.img_size}"
        
        # 1. Découpe l'image en patches (C, grid_size, patch_size, grid_size, patch_size)
        patches = img.unfold(1, self.patch_size, self.patch_size)\
                     .unfold(2, self.patch_size, self.patch_size)
        
        # 2. Reshape en liste de patches: (num_patches, C, patch_size, patch_size)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()
        patches = patches.view(self.num_patches, C, self.patch_size, self.patch_size)
        
        # 3. Applique la permutation
        patches_permuted = patches[self.permutation]
        
        # 4. Reconstruit l'image permutée
        img_permuted = self._reconstruct_image(patches_permuted)
        
        return img_permuted
    
    def _reconstruct_image(self, patches):
        """Reconstruit une image à partir de patches permutés"""
        C = patches.shape[1]
        
        # Reshape en grille: (grid_size, grid_size, C, patch_size, patch_size)
        patches_grid = patches.view(self.grid_size, self.grid_size, C, 
                                     self.patch_size, self.patch_size)
        
        # Reorganise les dimensions pour concaténer
        patches_grid = patches_grid.permute(2, 0, 3, 1, 4).contiguous()
        
        # Fusionne les patches: (C, H, W)
        img_reconstructed = patches_grid.view(C, self.img_size, self.img_size)
        
        return img_reconstructed
    
    def get_inverse_permutation(self):
        """Retourne la permutation inverse (pour visualisation)"""
        inverse = [0] * self.num_patches
        for i, p in enumerate(self.permutation):
            inverse[p] = i
        return inverse


# ============================================================================
# 2. Dataset & DataLoader
# ============================================================================

def download_imagenette(data_dir):
    """Télécharge et extrait le dataset Imagenette"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    extract_path = data_dir / "imagenette2"
    
    # Vérifie si déjà téléchargé
    if extract_path.exists():
        print(f"✅ Dataset Imagenette déjà présent dans {extract_path}")
        return extract_path
    
    print(f"📥 Téléchargement de Imagenette depuis {Config.DATASET_URL}...")
    
    # Télécharge
    tar_path = data_dir / "imagenette2.tgz"
    response = requests.get(Config.DATASET_URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(tar_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    print("📦 Extraction de l'archive...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(data_dir)
    
    # Nettoie l'archive
    tar_path.unlink()
    
    print(f"✅ Dataset Imagenette extrait dans {extract_path}")
    return extract_path


def create_dataloaders(use_puzzle=True):
    """
    Crée les DataLoaders pour train/val avec ou sans transformation puzzle.
    
    Args:
        use_puzzle: Si True, applique la permutation puzzle
    
    Returns:
        train_loader, val_loader, puzzle_transform
    """
    print("\n" + "="*70)
    print("📊 Création des DataLoaders")
    print("="*70)
    
    # Télécharge le dataset
    dataset_path = download_imagenette(Config.DATA_DIR)
    
    # Initialise la transformation puzzle
    puzzle_transform = PuzzlePermutation(
        img_size=Config.IMG_SIZE,
        patch_size=Config.PATCH_SIZE,
        seed=Config.SEED_PERMUTATION
    )
    
    # Transformations de base (sans puzzle)
    base_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    # Transformations complètes (avec puzzle si demandé)
    if use_puzzle:
        train_transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            puzzle_transform,  # Applique le puzzle AVANT la normalisation
            transforms.Normalize(Config.IMAGENET_MEAN, Config.IMAGENET_STD)
        ])
        val_transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            puzzle_transform,
            transforms.Normalize(Config.IMAGENET_MEAN, Config.IMAGENET_STD)
        ])
        print("🧩 Mode: PUZZLE ACTIVÉ")
    else:
        train_transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(Config.IMAGENET_MEAN, Config.IMAGENET_STD)
        ])
        val_transform = train_transform
        print("📷 Mode: IMAGES ORIGINALES")
    
    # Charge les datasets
    train_dataset = ImageFolder(dataset_path / "train", transform=train_transform)
    val_dataset = ImageFolder(dataset_path / "val", transform=val_transform)
    
    print(f"\n📈 Dataset Statistics:")
    print(f"   - Train: {len(train_dataset)} images")
    print(f"   - Val: {len(val_dataset)} images")
    print(f"   - Classes: {Config.NUM_CLASSES}")
    print(f"   - Class names: {train_dataset.classes}")
    
    # Crée les DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"\n🔄 DataLoader Info:")
    print(f"   - Batch size: {Config.BATCH_SIZE}")
    print(f"   - Train batches: {len(train_loader)}")
    print(f"   - Val batches: {len(val_loader)}")
    
    return train_loader, val_loader, puzzle_transform


# ============================================================================
# 3. Modèles (ResNet-50 vs ViT-Base/16)
# ============================================================================

def create_resnet50(num_classes=10, freeze_backbone=True):
    """
    Crée ResNet-50 pré-entraîné avec freezing optionnel du backbone.
    
    Stratégie de freezing:
    - Gèle toutes les couches convolutionnelles (feature extractor)
    - Entraîne uniquement la couche fc finale
    """
    print("\n🔨 Chargement de ResNet-50 (torchvision)...")
    
    # Charge le modèle pré-entraîné
    model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
    
    # Modifie la tête de classification pour 10 classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    if freeze_backbone:
        print("❄️  Freezing du backbone (Conv layers)...")
        # Gèle tous les paramètres sauf fc
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        
        # Compte les paramètres
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   ✓ Total params: {total_params:,}")
        print(f"   ✓ Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"   ✓ Frozen params: {total_params - trainable_params:,}")
    
    return model


def create_vit_base(num_classes=10, freeze_backbone=True):
    """
    Crée ViT-Base/16 pré-entraîné avec freezing optionnel du backbone.
    
    Stratégie de freezing:
    - Gèle tous les blocs Transformer (backbone)
    - Entraîne la tête de classification (head)
    - Entraîne les Positional Embeddings (pos_embed) ← CRUCIAL pour adapter la géométrie
    """
    print("\n🔨 Chargement de ViT-Base/16 (timm)...")
    
    # Charge le modèle pré-entraîné
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    
    if freeze_backbone:
        print("❄️  Freezing du backbone (Transformer blocks)...")
        # Gèle TOUS les paramètres d'abord
        for param in model.parameters():
            param.requires_grad = False
        
        # Dégèle uniquement:
        # 1. La tête de classification
        for param in model.head.parameters():
            param.requires_grad = True
        
        # 2. Les Positional Embeddings (LE CŒUR DE L'EXPÉRIENCE)
        if hasattr(model, 'pos_embed'):
            model.pos_embed.requires_grad = True
            print("   ✓ Positional Embeddings dégelés (adaptation géométrique)")
        
        # Compte les paramètres
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   ✓ Total params: {total_params:,}")
        print(f"   ✓ Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"   ✓ Frozen params: {total_params - trainable_params:,}")
    
    return model


# ============================================================================
# 4. Entraînement (Fine-tuning)
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Entraîne le modèle pour une époque"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Métriques
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Évalue le modèle sur le set de validation"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def train_model(model, model_name, train_loader, val_loader, num_epochs=5):
    """
    Entraîne un modèle et sauvegarde les métriques.
    
    Returns:
        history: Dict contenant les losses et accuracies
    """
    print("\n" + "="*70)
    print(f"🚀 Entraînement de {model_name}")
    print("="*70)
    
    model = model.to(Config.DEVICE)
    
    # Optimiseur (AdamW pour le fine-tuning)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Historique
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Entraînement
    for epoch in range(num_epochs):
        print(f"\n📍 Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)
        
        # Sauvegarde l'historique
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Affiche les résultats
        print(f"\n✅ Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Sauvegarde le modèle
    save_path = Config.RESULTS_DIR / f"{model_name}_final.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n💾 Modèle sauvegardé: {save_path}")
    
    return history


# ============================================================================
# 5. Visualisations
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
    axes[0].set_title('🎯 Validation Accuracy: ResNet vs ViT', fontsize=14, fontweight='bold')
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
        print(f"💾 Figure sauvegardée: {save_path}")
    
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



# ============================================================================
# 6. Script Principal
# ============================================================================

def main():
    """Fonction principale exécutant l'expérience complète"""
    
    print("\n" + "="*70)
    print("THE PUZZLE CHALLENGE: ResNet-50 vs ViT-Base/16")
    print("="*70)
    print("Objectif: Démontrer que le ViT s'adapte à la permutation spatiale")
    print("          alors que le CNN échoue (différence de biais inductif)")
    print("="*70 + "\n")
    
    # Configuration
    print(f"Configuration:")
    print(f"   - Device: {Config.DEVICE}")
    print(f"   - Image size: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
    print(f"   - Patch size: {Config.PATCH_SIZE}x{Config.PATCH_SIZE}")
    print(f"   - Batch size: {Config.BATCH_SIZE}")
    print(f"   - Epochs: {Config.NUM_EPOCHS}")
    print(f"   - Learning rate: {Config.LEARNING_RATE}")
    
    # Crée les dossiers de résultats
    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Étape 1: Charge les données AVEC la transformation puzzle
    # ========================================================================
    train_loader, val_loader, puzzle_transform = create_dataloaders(use_puzzle=True)
    
    # ========================================================================
    # Étape 2: Visualise l'effet du puzzle
    # ========================================================================
    print("\n" + "="*70)
    print("Visualisation de la Transformation Puzzle")
    print("="*70)
    
    # Crée un dataset temporaire sans transformations pour la visualisation
    dataset_path = Config.DATA_DIR / "imagenette2"
    val_dataset_viz = ImageFolder(dataset_path / "val", transform=None)
    
    visualize_puzzle_effect(
        puzzle_transform,
        val_dataset_viz,
        num_samples=4,
        save_path=Config.RESULTS_DIR / "puzzle_effect.png"
    )
    
    # ========================================================================
    # Étape 3: Entraîne ResNet-50
    # ========================================================================
    resnet = create_resnet50(num_classes=Config.NUM_CLASSES, freeze_backbone=True)
    history_resnet = train_model(
        resnet,
        "ResNet50_Puzzle",
        train_loader,
        val_loader,
        num_epochs=Config.NUM_EPOCHS
    )
    
    # ========================================================================
    # Étape 4: Entraîne ViT-Base/16
    # ========================================================================
    vit = create_vit_base(num_classes=Config.NUM_CLASSES, freeze_backbone=True)
    history_vit = train_model(
        vit,
        "ViT_Base_Puzzle",
        train_loader,
        val_loader,
        num_epochs=Config.NUM_EPOCHS
    )
    
    # ========================================================================
    # Étape 5: Compare les résultats
    # ========================================================================
    print("\n" + "="*70)
    print("📊 Comparaison des Résultats")
    print("="*70)
    
    plot_training_comparison(
        history_resnet,
        history_vit,
        save_path=Config.RESULTS_DIR / "training_comparison.png"
    )
    
    # Affiche les résultats finaux
    print("\n🏆 Résultats Finaux (Époque {}):\n".format(Config.NUM_EPOCHS))
    print("┌─────────────────┬──────────────┬──────────────┐")
    print("│ Modèle          │ Val Accuracy │ Val Loss     │")
    print("├─────────────────┼──────────────┼──────────────┤")
    print(f"│ ResNet-50       │ {history_resnet['val_acc'][-1]:>11.2f}% │ {history_resnet['val_loss'][-1]:>12.4f} │")
    print(f"│ ViT-Base/16     │ {history_vit['val_acc'][-1]:>11.2f}% │ {history_vit['val_loss'][-1]:>12.4f} │")
    print("└─────────────────┴──────────────┴──────────────┘")
    
    # Sauvegarde les historiques
    import json
    with open(Config.RESULTS_DIR / "history_resnet.json", 'w') as f:
        json.dump(history_resnet, f, indent=2)
    with open(Config.RESULTS_DIR / "history_vit.json", 'w') as f:
        json.dump(history_vit, f, indent=2)
    
    print(f"\n💾 Historiques sauvegardés dans {Config.RESULTS_DIR}")


if __name__ == "__main__":
    # Set random seeds pour la reproductibilité
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Lance l'expérience
    main()
