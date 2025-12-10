"""
Binary classification models 
"""
import torch
import torch.nn as nn

class ClassificationNetwork(nn.Module):
    def __init__(self):
        super(ClassificationNetwork, self).__init__()
        
        # Blocs convolutifs
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Flatten
        self.flatten = nn.Flatten()

        # Si entrée = 256x256 : après 4 maxpool -> 16x16, channels=128
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU()
        )
        
        self.dropout = nn.Dropout(0.5)

        # 1 seule sortie pour classification binaire 
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x
    

class MyVGGClassifier(nn.Module):
    def __init__(self, conv_base, num_classes=2):
        super().__init__()
        self.conv_base = conv_base
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))  # comme dans VGG16
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512*7*7, 1)   
        
    def forward(self, x):
        x = self.conv_base(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x