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

# Normalisation officielle ImageNet
MEAN = [0.485, 0.456, 0.406] 
STD = [0.229, 0.224, 0.225]

def get_train_loader(data_dir=TRAIN_DATA_DIR, batch_size=BATCH_SIZE):
    """
    Crée un DataLoader pour l'entraînement (photos réelles de mini-ImageNet).
    Inclut des augmentations de données.
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    
    print(f"✅ Dataset d'ENTRAÎNEMENT chargé depuis : {data_dir}")
    print(f"   - Nombre d'images : {len(dataset)}")
    print(f"   - Classes trouvées : {dataset.classes}")

    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, dataset.classes

def get_val_loader(data_dir=VAL_DATA_DIR, batch_size=BATCH_SIZE):
    """
    Crée un DataLoader pour la validation (photos réelles de mini-ImageNet).
    """
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)
    
    print(f"✅ Dataset de VALIDATION chargé depuis : {data_dir}")
    print(f"   - Nombre d'images : {len(dataset)}")
    print(f"   - Classes trouvées : {dataset.classes}")

    val_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return val_loader, dataset.classes

def get_test_loader(data_dir=TEST_DATA_DIR, batch_size=BATCH_SIZE):
    """
    Crée un DataLoader pour le test (images stylisées d'ImageNet-R).
    """
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), 
        transforms.ToTensor(),                   
        transforms.Normalize(mean=MEAN, std=STD) 
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=test_transform)
    
    print(f" Dataset de TEST chargé depuis : {data_dir}")
    print(f"   - Nombre d'images : {len(dataset)}")
    print(f"   - Classes trouvées : {dataset.classes}")

    test_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return test_loader, dataset.classes

def imshow_batch(dataloader, class_names):
    """
    Affiche 4 images d'un batch pour vérifier que tout va bien.
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

# --- CORRECTION ICI : La fonction est sortie (désindentée) ---
def imshow_batch_by_class(dataloader, class_names, target_class, offset=0, n_images=4):
    """
    Affiche n_images d'une classe spécifique du dataloader de test.
    - target_class : nom de la classe (str) ou index (int)
    - offset : permet de ne pas toujours voir les mêmes images (décalage dans la liste)
    - n_images : nombre d'images à afficher
    """
    # Trouver l'index de la classe si string
    if isinstance(target_class, str):
        try:
            class_idx = class_names.index(target_class)
        except ValueError:
            print(f"❌ Erreur : La classe '{target_class}' n'existe pas dans la liste.")
            return
    else:
        class_idx = target_class

    # Parcourir le dataset pour trouver les images de la classe
    dataset = dataloader.dataset
    # Note : dataset.samples contient des tuples (chemin, index_label)
    # On cherche tous les indices qui correspondent à notre classe
    indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
    
    if not indices:
        print(f"Aucune image trouvée pour la classe '{class_names[class_idx]}'")
        return

    # Vérifier l'offset
    if offset >= len(indices):
        print(f"⚠️ Offset trop grand ({offset}). Il n'y a que {len(indices)} images pour cette classe.")
        return

    # Sélectionner les indices voulus
    selected_indices = indices[offset:offset+n_images]
    
    images = []
    labels = []
    
    # Charger les images une par une
    for idx in selected_indices:
        img, label = dataset[idx]
        images.append(img)
        labels.append(label)

    # Affichage
    n_cols = len(images)
    if n_cols == 0:
        return

    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 5))
    
    # Si une seule image, axes n'est pas une liste, on le transforme
    if n_cols == 1:
        axes = [axes]
        
    for i, (img, label) in enumerate(zip(images, labels)):
        img_np = img.numpy().transpose((1, 2, 0))
        mean = np.array(MEAN)
        std = np.array(STD)
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        axes[i].imshow(img_np)
        axes[i].set_title(f"{class_names[label]}\nIndex: {selected_indices[i]}")
        axes[i].axis("off")
        
    plt.tight_layout()
    plt.show()

# --- TEST  ---
if __name__ == "__main__":
    loader, classes = get_test_loader()
    imshow_batch(loader, classes)