## We import here ResNet50 and vit_b_16 from torchvision.models

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn as nn

def get_resnet50_model(num_classes):
    # Utiliser les poids par défaut (IMAGENET1K_V1 ou V2)
    weights = ResNet50_Weights.DEFAULT 
    model = resnet50(weights=weights)
    
    # Remplacer la dernière couche
    # On récupère la taille d'entrée (2048 pour ResNet50)
    in_features = model.fc.in_features 
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

def get_vit_b_16_model(num_classes):
    # Utiliser les poids par défaut (souvent entraînés sur ImageNet-1K avec recette améliorée)
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    
    # Remplacer la tête
    # Structure torchvision : model.heads.head
    in_features = model.heads.head.in_features # 768 pour ViT-B/16
    model.heads.head = nn.Linear(in_features, num_classes)
    
    return model