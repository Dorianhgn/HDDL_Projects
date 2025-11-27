"""
Classification models for bounding box regression.

Options:
1. UNet encoder + regression head
2. VGG/ResNet + regression head with different freezing strategies
"""

import torch
import torch.nn as nn
import torchvision.models as models
from models.unet import Encoder


class UNetEncoderRegressor(nn.Module):
    """
    Option 1: Extract UNet encoder and attach a regression head.
    Uses pre-trained UNet weights if provided.
    """
    def __init__(self, pretrained_path=None, C_in=3, C_hid=64, n_s=4, num_outputs=4):
        super().__init__()
        
        # Load encoder from UNet
        self.encoder = Encoder(C_in, C_hid, n_s)
        
        # Load pretrained weights if available
        if pretrained_path:
            print(f"Loading UNet encoder weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            # Extract only encoder weights
            encoder_state = {k.replace('encoder.', ''): v 
                           for k, v in checkpoint.items() 
                           if k.startswith('encoder.')}
            self.encoder.load_state_dict(encoder_state, strict=False)
        
        # Global average pooling to reduce spatial dimensions
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Regression head
        final_channels = self.encoder.final_channels
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs)
        )
    
    def forward(self, x):
        # Encode
        x_final, _ = self.encoder(x)
        
        # Global pooling
        x = self.gap(x_final)
        
        # Regression head
        x = self.head(x)
        
        return x


class VGGRegressor(nn.Module):
    """
    Option 2: VGG16 + regression head with configurable freezing.
    
    Args:
        freeze_strategy: 'none', 'head_only', 'head_and_last_conv', 'all_except_head'
        pretrained: Use ImageNet pre-trained weights
        num_outputs: Number of output values (4 for bounding boxes)
    """
    def __init__(self, freeze_strategy='none', pretrained=True, num_outputs=4):
        super().__init__()
        
        # Load VGG16 with batch norm (more stable)
        self.backbone = models.vgg16_bn(pretrained=pretrained)
        
        # Remove the classifier
        self.features = self.backbone.features
        
        # Get the number of features (512 for VGG16)
        num_features = 512
        
        # Regression head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(num_features * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_outputs)
        )
        
        self._apply_freeze_strategy(freeze_strategy)
    
    def _apply_freeze_strategy(self, strategy):
        """Apply freezing strategy to model parameters."""
        if strategy == 'none':
            # Train everything (full fine-tuning)
            pass
        
        elif strategy == 'head_only':
            # Freeze all backbone, train only head
            for param in self.features.parameters():
                param.requires_grad = False
        
        elif strategy == 'head_and_last_conv':
            # Freeze all except head and last conv block
            # VGG16 has 5 conv blocks, we unfreeze the last one (indices 43+)
            for i, param in enumerate(self.features.parameters()):
                if i < 43:  # Freeze first 4 blocks
                    param.requires_grad = False
        
        elif strategy == 'all_except_head':
            # Same as head_only
            for param in self.features.parameters():
                param.requires_grad = False
        
        else:
            raise ValueError(f"Unknown freeze_strategy: {strategy}")
        
        # Print trainable parameters
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"VGG16 - Strategy: {strategy}")
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x


class ResNetRegressor(nn.Module):
    """
    Option 2: ResNet18/50 + regression head with configurable freezing.
    
    Args:
        architecture: 'resnet18' or 'resnet50'
        freeze_strategy: 'none', 'head_only', 'head_and_last_conv', 'all_except_head'
        pretrained: Use ImageNet pre-trained weights
        num_outputs: Number of output values (4 for bounding boxes)
    """
    def __init__(self, architecture='resnet18', freeze_strategy='none', 
                 pretrained=True, num_outputs=4):
        super().__init__()
        
        # Load ResNet
        if architecture == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = 512
        elif architecture == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = 2048
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Remove the final fc layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Regression head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs)
        )
        
        self._apply_freeze_strategy(freeze_strategy)
    
    def _apply_freeze_strategy(self, strategy):
        """Apply freezing strategy to model parameters."""
        if strategy == 'none':
            # Train everything (full fine-tuning)
            pass
        
        elif strategy == 'head_only':
            # Freeze all backbone, train only head
            for param in self.features.parameters():
                param.requires_grad = False
        
        elif strategy == 'head_and_last_conv':
            # Freeze all except head and last residual block (layer4)
            modules = list(self.backbone.children())[:-1]  # Remove avgpool
            for module in modules[:-1]:  # Freeze conv1, bn1, relu, maxpool, layer1-3
                for param in module.parameters():
                    param.requires_grad = False
        
        elif strategy == 'all_except_head':
            # Same as head_only
            for param in self.features.parameters():
                param.requires_grad = False
        
        else:
            raise ValueError(f"Unknown freeze_strategy: {strategy}")
        
        # Print trainable parameters
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"ResNet ({strategy}) - Strategy: {strategy}")
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x
