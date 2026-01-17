import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os

class CLEVR_ReasoningDataset(Dataset):
    def __init__(self, json_file, img_root_dir, transform=None):
        """
        json_file: chemin vers 'task1_train.json' ou 'task2_train.json'
        img_root_dir: chemin vers le dossier contenant les images (ex: .../images/train)
        """
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.img_root_dir = img_root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_name = item['image']
        label = item['label']
        
        # Construction du chemin image
        img_path = os.path.join(self.img_root_dir, img_name)
        
        # Chargement image (Convertir en RGB pour éviter les soucis de canal alpha)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    # --- EXEMPLE D'UTILISATION ---

    # 1. Définir les transformations (Vital pour ViT : resize 224x224 + Norm ImageNet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # ViT a besoin de patchs fixes
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Instancier le Dataset pour la Tâche 2 (Sphère Grise)
    train_dataset = CLEVR_ReasoningDataset(
        json_file="task2_train.json",
        img_root_dir="data_clevr/CLEVR_v1.0/images/train",
        transform=transform
    )

    # 3. Créer le DataLoader
    # num_workers=4 est standard sur un cluster, pin_memory=True pour GPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    # Test rapide
    print(f"Nombre d'images Tâche 2 : {len(train_dataset)}")
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")