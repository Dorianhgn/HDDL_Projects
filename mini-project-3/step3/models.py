import torch.nn as nn
import torchvision
import timm


# ============================================================================
# Modèles (ResNet-50 vs ViT-Base/16)
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