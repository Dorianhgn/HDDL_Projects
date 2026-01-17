import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

# --- CONFIGURATION STANDARD ---
# Obtenir le chemin du fichier dataloader.py
CURRENT_FILE = Path(__file__).resolve()
STEP4_DIR = CURRENT_FILE.parent  # Dossier step4

# Chemins des datasets
TRAIN_DATA_DIR = STEP4_DIR / "data" / "mini-imagenet-train"  # Photos réelles pour entraînement
VAL_DATA_DIR = STEP4_DIR / "data" / "mini-imagenet-val"      # Photos réelles pour validation
TEST_DATA_DIR = STEP4_DIR / "data" / "imagenet-r-mini"       # Images stylisées pour test

BATCH_SIZE = 32
IMG_SIZE = 224  # Taille standard pour ConvNeXt et Swin
NUM_WORKERS = 2 # Pour charger rapidement 

# Normalisation officielle ImageNet (CRITIQUE pour que les modèles SOTA fonctionnent)
MEAN = [0.485, 0.456, 0.406] 
STD = [0.229, 0.224, 0.225]
# C'est la moyenne et l'écart-type de chaque pixel (Rouge, Vert, Bleu) calculés sur les 1,2 million d'images du dataset ImageNet original.
# On normalise par les moyennes, écarts types du jeu train ImageNet car les modèles pré-entraînés ont été entraînés avec cette normalisation.

def get_train_loader(data_dir=TRAIN_DATA_DIR, batch_size=BATCH_SIZE):
    """
    Crée un DataLoader pour l'entraînement (photos réelles de mini-ImageNet).
    Inclut des augmentations de données.
    """
    
    # Transformations d'entraînement avec augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    # Charger le dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    
    print(f"✅ Dataset d'ENTRAÎNEMENT chargé depuis : {data_dir}")
    print(f"   - Nombre d'images : {len(dataset)}")
    print(f"   - Classes trouvées : {dataset.classes}")

    # Créer le DataLoader
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,            # Mélanger pour l'entraînement
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, dataset.classes

def get_val_loader(data_dir=VAL_DATA_DIR, batch_size=BATCH_SIZE):
    """
    Crée un DataLoader pour la validation (photos réelles de mini-ImageNet).
    Transformations simples sans augmentation.
    """
    
    # Transformations de validation (sans augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    # Charger le dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)
    
    print(f"✅ Dataset de VALIDATION chargé depuis : {data_dir}")
    print(f"   - Nombre d'images : {len(dataset)}")
    print(f"   - Classes trouvées : {dataset.classes}")

    # Créer le DataLoader
    val_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,           # Pas de mélange pour la validation
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return val_loader, dataset.classes

def get_test_loader(data_dir=TEST_DATA_DIR, batch_size=BATCH_SIZE):
    """
    Crée un DataLoader pour le test (images stylisées d'ImageNet-R).
    Transformations simples sans augmentation.
    """
    
    # 1. Transformations simples (Juste mise à l'échelle et normalisation)
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # Redimensionner à 224x224
        transforms.ToTensor(),                   # Convertir en Tensor (0-1)
        transforms.Normalize(mean=MEAN, std=STD) # Normaliser (Maths)
    ])

    # 2. Charger le dossier
    # ImageFolder associe automatiquement les dossiers aux classes
    dataset = datasets.ImageFolder(root=data_dir, transform=test_transform)
    
    print(f" Dataset de TEST chargé depuis : {data_dir}")
    print(f"   - Nombre d'images : {len(dataset)}")
    print(f"   - Classes trouvées : {dataset.classes}")

    # 3. Créer le DataLoader (Le Serveur)
    test_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,           # On ne mélange pas pour garder l'ordre des images
        num_workers=NUM_WORKERS,
        pin_memory=True         # Accélérateur GPU
    )

    return test_loader, dataset.classes

def imshow_batch(dataloader, class_names):
    """
    Affiche 4 images d'un batch pour vérifier que tout va bien.
    Gère la dé-normalisation pour l'affichage humain.
    """
    # Récupérer un batch
    images, labels = next(iter(dataloader))
    
    # Garder seulement 4 images
    images = images[:4]
    labels = labels[:4]

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    
    for idx, image in enumerate(images):
        # Dé-normaliser : (image * std) + mean
        image = image.numpy().transpose((1, 2, 0)) # Changer l'ordre : (C,H,W) -> (H,W,C)
        mean = np.array(MEAN)
        std = np.array(STD)
        image = std * image + mean
        image = np.clip(image, 0, 1) # Corriger les petits débordements de pixels
        
        axes[idx].imshow(image)
        axes[idx].set_title(class_names[labels[idx]])
        axes[idx].axis("off")
    
    plt.show()

# --- TEST  ---
if __name__ == "__main__":
    loader, classes = get_test_loader()
    imshow_batch(loader, classes)