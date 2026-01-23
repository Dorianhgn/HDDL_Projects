import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataloader_classification import get_oxford_loaders

# Configuration
root_dir = './data/oxford-iiit-pet'
batch_size = 16
classification_type = 'multiclass'  # <-- multiclass par races
num_epochs = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) Créer les dataloaders
loaders = get_oxford_loaders(root_dir, classification_type=classification_type, batch_size=batch_size)
train_loader = loaders['train']
val_loader = loaders['val']

dataset = train_loader.dataset
num_classes = len(dataset.label_to_idx)
print(f"Num classes (mapping size): {num_classes}")
print("Some labels (first 10):", dict(list(dataset.label_to_idx.items())[:10]))

# 2) Modèle: ResNet18 pré-entraîné, adapter la dernière couche
model = models.resnet18(weights=None)
# remplacer la dernière fc
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)

# 3) Critère et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# 4) Boucle d'entraînement minimale (1 epoch)
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i+1}/{len(train_loader)}] Loss: {running_loss / 10:.4f}")
            running_loss = 0.0

print('Training finished (minimal run).')

# 5) Sauvegarder un checkpoint rapide
torch.save({'model_state_dict': model.state_dict(), 'label_to_idx': dataset.label_to_idx}, 'resnet18_multiclass_checkpoint.pth')
print('Checkpoint saved to resnet18_multiclass_checkpoint.pth')
