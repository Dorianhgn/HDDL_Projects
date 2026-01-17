import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import argparse
import os
import numpy as np
from tqdm import tqdm

# Imports locaux
from dataloader import CLEVR_ReasoningDataset
from models import get_resnet50_model, get_vit_b_16_model

def get_model(model_name, num_classes):
    if model_name == 'resnet50':
        return get_resnet50_model(num_classes)
    elif model_name == 'vit_b_16':
        return get_vit_b_16_model(num_classes)
    else:
        raise ValueError(f"Unknown model {model_name}")

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return total_loss / len(loader), 100 * correct / total

def train(args):
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model} | Task: {args.task}")
    
    exp_dir = os.path.join("experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # --- 2. Data Preparation ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # On charge TOUT le dataset d'entraînement original
    json_train_path = f"task{args.task}_train.json"
    full_dataset = CLEVR_ReasoningDataset(
        json_file=json_train_path,
        img_root_dir=args.img_dir, # Dossier images/train
        transform=transform
    )

    # Split dynamique : 80% Train, 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Data split: {train_size} Train samples | {val_size} Val samples")

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- 3. Model & Optimization ---
    # CLEVR colors = 8 classes
    model = get_model(args.model, num_classes=8).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- 4. Training Loop ---
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(acc=100 * correct / total, loss=loss.item())

        # Validation Step
        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = 100 * correct / total
        avg_val_loss, avg_val_acc = evaluate(model, val_loader, device, criterion)

        print(f" -> Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.2f}%")

        # Logging
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        np.save(os.path.join(exp_dir, "metrics.npy"), history)

        # Checkpointing
        torch.save(model.state_dict(), os.path.join(exp_dir, "last.pth"))
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))
            print(" -> New Best Model Saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Folder name for outputs")
    parser.add_argument("--task", type=int, required=True, choices=[1, 2], help="1=Cylinders, 2=Sphere")
    parser.add_argument("--model", type=str, required=True, choices=['resnet50', 'vit_b_16'])
    parser.add_argument("--img_dir", type=str, default="data_clevr/CLEVR_v1.0/images/train")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    args = parser.parse_args()
    train(args)