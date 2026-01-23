"""
Configuration centralisée pour l'expérience Puzzle Challenge
"""

import torch
from pathlib import Path


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
    NUM_EPOCHS = 10  # Augmenté à 10 époques
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ImageNet Stats
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
