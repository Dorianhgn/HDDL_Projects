import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet50Pets(nn.Module):
    def __init__(self, num_classes=37, freeze_backbone=True):
        super(ResNet50Pets, self).__init__()
        
        # Chargement du modèle pré-entraîné sur ImageNet
        self.resnet = models.resnet50(pretrained=True)
        
        # Option : Geler les couches convolutionnelles
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
                
        # ResNet50 : 2048 features en sortie du backbone
        num_features = self.resnet.fc.in_features  # 2048
        self.resnet.fc = nn.Sequential(
            # Première couche : 2048 → 512
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            
            # Deuxième couche : 512 → 256
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            # Couche de sortie : 256 → num_classes
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)
    
    def unfreeze_backbone(self):
        """Permet de dégeler progressivement le backbone"""
        for param in self.resnet.parameters():
            param.requires_grad = True
    
    def freeze_backbone(self):
        """Gèle à nouveau le backbone (utile pour debug)"""
        for name, param in self.resnet.named_parameters():
            if 'fc' not in name:  # Ne gèle pas la tête
                param.requires_grad = False

class SimplePetCNN(nn.Module):
    def __init__(self, num_classes=37):
        super(SimplePetCNN, self).__init__()
        
        # Bloc 1 : Entrée (3, H, W) -> Sortie (32, H/2, W/2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloc 2 : (32, H/2, W/2) -> (64, H/4, W/4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Bloc 3 : (64, H/4, W/4) -> (128, H/8, W/8)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Bloc 4 : (128, H/8, W/8) -> (256, H/16, W/16)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pour 256x256 input: après 4 maxpool -> 16x16
        # 256 * 16 * 16 = 65536
        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        
        x = torch.flatten(x, 1)    # (batch_size, 256*16*16)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x