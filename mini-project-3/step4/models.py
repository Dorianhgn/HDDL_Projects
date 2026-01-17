
import torch.nn as nn
from torchvision.models import (
    convnext_tiny, ConvNeXt_Tiny_Weights,
    swin_t, Swin_T_Weights
)

def get_convnext_model(num_classes):
    """
    Charge un ConvNeXt-Tiny pré-entraîné et adapte la dernière couche.
    """
    print(" Chargement de ConvNeXt-Tiny (SOTA CNN)...")
    
    # 1. Charger les poids (IMAGENET1K_V1)
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    model = convnext_tiny(weights=weights)
    
    # 2. Remplacer la tête de classification
    # Dans ConvNeXt, la tête est une séquence : (LayerNorm, Flatten, Linear)
    # La couche linéaire est la dernière (index 2)
    in_features = model.classifier[2].in_features
    
    # On remplace uniquement cette couche linéaire
    model.classifier[2] = nn.Linear(in_features, num_classes)
    
    return model

def get_swin_model(num_classes):
    """
    Charge un Swin Transformer-Tiny pré-entraîné et adapte la dernière couche.
    """
    print(" Chargement de Swin Transformer-Tiny (SOTA ViT)...")
    
    # 1. Charger les poids
    weights = Swin_T_Weights.DEFAULT
    model = swin_t(weights=weights)
    
    # 2. Remplacer la tête de classification
    # Dans Swin, c'est directement 'head'
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    
    return model
# il faut réentrainer les nouvelles têtes sur notre dataset cible après avoir chargé les modèles pré-entraînés parceque les têtes originales sont adaptées aux 1000 classes d'ImageNet, pas aux 10 classes de notre dataset.