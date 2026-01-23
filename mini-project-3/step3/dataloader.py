import random
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from pathlib import Path
import requests
import tarfile
from PIL import Image

from config import Config


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
        
        print(f"🧩 Puzzle Permutation initialisée:")
        print(f"   - Image size: {img_size}x{img_size}")
        print(f"   - Patch size: {patch_size}x{patch_size}")
        print(f"   - Grid: {self.grid_size}x{self.grid_size} = {self.num_patches} patches")
        print(f"   - Seed: {seed} (permutation fixe)")
    
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