import dask.dataframe as dd
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import io  # Pour convertir le string en dictionnaire
import ast  # Pour convertir le string en dictionnaire

# Chargement du mapping
try:
    mapping_df = pd.read_csv("mini-imagenet_imagenet-r_crossed_classes.csv")
# except FileNotFoundError:
#     mapping_df = pd.read_csv("step4/mini-imagenet_imagenet-r_crossed_classes.csv")
except FileNotFoundError:
    mapping_df = pd.read_csv("step4_without_download/mini-imagenet_imagenet-r_crossed_classes.csv")

MINI_LABEL_TO_IDX = {int(row['label']): i for i, row in mapping_df.iterrows()}
WNID_TO_IDX = {row['wnid']: i for i, row in mapping_df.iterrows()}

TARGET_MINI_LABELS = list(MINI_LABEL_TO_IDX.keys())
TARGET_WNIDS = list(WNID_TO_IDX.keys())

class ParquetDataset(Dataset):
    def __init__(self, hf_path, split_pattern, target_list, mode="mini", transform=None):
        full_path = f"hf://datasets/{hf_path}/{split_pattern}"
        ddf = dd.read_parquet(full_path)
        
        col_name = 'label' if mode == "mini" else 'wnid'
        # Filtrage lazy
        filtered_ddf = ddf[ddf[col_name].isin(target_list)]
        
        # Passage en Pandas pour l'accès rapide par index
        self.df = filtered_ddf.compute().reset_index(drop=True)
        self.transform = transform
        self.mode = mode
        self.col_name = col_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- Extraction de l'image (Gestion du string-dict) ---
        img_data = row['image']
        
        # Si c'est un string, on le convertit en dict
        if isinstance(img_data, str):
            img_data = ast.literal_eval(img_data)
        
        # On extrait les bytes
        img_bytes = img_data['bytes']
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # --- Mapping du Label ---
        raw_label = row[self.col_name]
        label_idx = MINI_LABEL_TO_IDX[int(raw_label)] if self.mode == "mini" else WNID_TO_IDX[str(raw_label)]
            
        if self.transform:
            image = self.transform(image)
            
        return image, label_idx

def get_dataloaders(batch_size=32, num_workers=4):
    # Transforms standards ImageNet
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    # Datasets
    print("Chargement Mini-ImageNet Train...")
    train_ds = ParquetDataset("timm/mini-imagenet", "data/train-*.parquet", 
                              TARGET_MINI_LABELS, mode="mini", transform=train_transform)
    
    print("Chargement Mini-ImageNet Val...")
    val_ds = ParquetDataset("timm/mini-imagenet", "data/validation-*.parquet", 
                            TARGET_MINI_LABELS, mode="mini", transform=test_transform)
    
    print("Chargement ImageNet-R Test...")
    test_ds = ParquetDataset("axiong/imagenet-r", "test/test-*.parquet", 
                             TARGET_WNIDS, mode="r", transform=test_transform)

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader