"""
Loss functions pour la segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss pour la segmentation multi-classes
    
    Meilleure que CrossEntropy pour :
    - Gérer le déséquilibre de classes
    - Optimiser directement l'IoU/Dice Score
    - Améliorer la précision des bords
    """
    def __init__(self, n_classes=3, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [Batch, n_classes, H, W] (logits avant softmax)
            targets: [Batch, H, W] (indices de classes 0, 1, 2)
        
        Returns:
            loss: scalaire entre 0 et 1 (0 = parfait)
        """
        # Softmax pour obtenir des probabilités
        inputs = F.softmax(inputs, dim=1)  # [B, C, H, W]
        
        # One-hot encoding des targets
        targets_one_hot = F.one_hot(targets, self.n_classes)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        # Calcul du Dice par classe
        # Intersection = somme des produits élément par élément
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))  # [B, C]
        
        # Union = somme des prédictions + somme des targets
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))  # [B, C]
        
        # Dice coefficient par classe
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [B, C]
        
        # Dice Loss = 1 - moyenne des dice
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combinaison de CrossEntropy et Dice Loss
    
    - CrossEntropy : pour la stabilité de l'entraînement
    - Dice Loss : pour la précision de la segmentation
    """
    def __init__(self, n_classes=3, ce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(n_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.ce_weight * ce + self.dice_weight * dice


class FocalLoss(nn.Module):
    """
    Focal Loss pour gérer le déséquilibre de classes
    
    Utile pour les bords (classe minoritaire)
    """
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [Batch, n_classes, H, W]
            targets: [Batch, H, W]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
