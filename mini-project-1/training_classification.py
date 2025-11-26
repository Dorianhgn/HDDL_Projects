import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm 
import os
from models.unet import UNet
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train UNet for segmentation')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_s', type=int, default=3, help='Number of scales')
    parser.add_argument('--n_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--path', type=str, default='models/unet_ns3/', help='Path to save models')
    parser.add_argument('--continue_training', action='store_true', help='Continue from last checkpoint')
    parser.add_argument('--data_path', type=str, default='data/oxford-iiit-pet', help='Path to dataset')
    return parser.parse_args()


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    for images, masks in tqdm(loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_val = 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_val += loss.item()
    
    return running_val / len(loader)


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entraînement lancé sur : {device}")
    
    # Paths
    best_model_path = os.path.join(args.path, "best.pth")
    last_model_path = os.path.join(args.path, "last.pth")
    loss_log_path = os.path.join(args.path, "loss_log.npy")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    if not os.path.exists(args.path):
        os.makedirs(args.path)
    
    # Load data
    from dataloader_segmentation import get_oxford_loaders
    loaders = get_oxford_loaders(args.data_path, task='contours', batch_size=args.batch_size)
    
    # Initialize model
    model = UNet(n_classes=args.n_classes, n_s=args.n_s).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Checkpoint logic
    min_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    if args.continue_training and os.path.exists(last_model_path):
        print(f"🔄 Reprise de l'entraînement depuis : {last_model_path}")
        model.load_state_dict(torch.load(last_model_path))
        
        print("   Calcul du score actuel du modèle chargé...")
        min_val_loss = validate(model, loaders['val'], criterion, device)
        print(f"   💪 Le modèle reprend avec une Val Loss de référence de : {min_val_loss:.4f}")
        
        if os.path.exists(loss_log_path):
            loss_log = np.load(loss_log_path, allow_pickle=True).item()
            train_losses = loss_log.get('train', [])
            val_losses = loss_log.get('val', [])
            print(f"   📈 Courbes de perte chargées avec {len(train_losses)} époques précédentes.")
    else:
        print("✨ Démarrage d'un nouvel entraînement (de zéro).")
    
    print(f"\nC'est parti pour {args.epochs} époques !")
    
    # Training loop
    for epoch in range(args.epochs):
        avg_train = train_epoch(model, loaders['train'], criterion, optimizer, device)
        train_losses.append(avg_train)
        
        avg_val = validate(model, loaders['val'], criterion, device)
        val_losses.append(avg_val)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train:.4f} - Val Loss: {avg_val:.4f}")
        
        # Save best model
        if avg_val < min_val_loss:
            diff = min_val_loss - avg_val
            torch.save(model.state_dict(), best_model_path)
            min_val_loss = avg_val
            print(f"   💾 Amélioration (-{diff:.4f}) -> Sauvegarde (Val Loss: {avg_val:.4f})")
        else:
            print(f"   (Val Loss: {avg_val:.4f}) - Pas d'amélioration")
        
        # Save last model
        torch.save(model.state_dict(), last_model_path)
        
        # Save loss log
        loss_log = {'train': train_losses, 'val': val_losses}
        np.save(loss_log_path, loss_log)
    
    print("✅ Fin de l'entraînement !")


if __name__ == '__main__':
    main()

