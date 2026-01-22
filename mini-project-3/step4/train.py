"""
Script d'entraînement (fine-tuning) pour ConvNeXt et Swin Transformer.

Pipeline :
1. Train sur Mini-ImageNet (photos réelles) - 21 classes
2. Validation sur Mini-ImageNet validation
3. Sauvegarde des meilleurs poids

"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import argparse

from dataloader import get_train_loader, get_val_loader
from models import get_convnext_model, get_swin_model

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 21

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """
    Entraîne le modèle pour une epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistiques
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Mise à jour de la barre de progression
        pbar.set_postfix({
            'loss': f'{running_loss/len(train_loader):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """
    Évalue le modèle sur le jeu de validation (mini-ImageNet val).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def train_model(model, model_name, train_loader, val_loader, epochs, learning_rate, save_dir):
    """
    Pipeline d'entraînement complet.
    """
    print(f"\n{'='*60}")
    print(f"  ENTRAÎNEMENT : {model_name}")
    print(f"{'='*60}\n")
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {epochs}")
    print(f"  Learning Rate : {learning_rate}")
    print(f"  Nombre de classes : {NUM_CLASSES}\n")
    
    model = model.to(DEVICE)
    
    # Loss et optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Historique
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
    # Créer le dossier de sauvegarde
    SAVE_DIR_PATH = save_dir
    SAVE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    
    # Boucle d'entraînement
    for epoch in range(epochs):
        # Entraînement
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch, epochs
        )
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # Learning rate scheduler
        scheduler.step()
        
        # Sauvegarder l'historique
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        np.save(os.path.join(SAVE_DIR_PATH, "metrics.npy"), history)
        
        # Afficher les résultats
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Sauvegarder le meilleur modèle (accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = SAVE_DIR_PATH / f"{model_name}_best.pth"
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, checkpoint_path)
                print(f"  ✓ Meilleur modèle sauvegardé : {checkpoint_path}")
            except Exception as e:
                print(f"    Erreur sauvegarde: {e}")
                # Essayer sans optimizer (plus léger)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, checkpoint_path)
                print(f"  ✓ Meilleur modèle sauvegardé (sans optimizer)")
        
        # Sauvegarder meilleur modèle (loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = SAVE_DIR_PATH / f"{model_name}_best_loss.pth"
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, checkpoint_path)
                print(f"  ✓ Meilleur modèle (loss) sauvegardé : {checkpoint_path}")
            except:
                pass  # Ignorer si échec
    
    print(f"\n Entraînement terminé !")
    print(f"  Meilleure précision sur validation : {best_val_acc:.2f}%")
    print(f"  Checkpoints sauvegardés dans : {SAVE_DIR_PATH}\n")
    
    return history, best_val_acc

def plot_training_history(history, model_name, save_dir):
    """
    Affiche les courbes d'entraînement.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = save_dir / f"{model_name}_training.png"
    plt.savefig(plot_path, dpi=150)
    print(f"   Courbes sauvegardées : {plot_path}")
    plt.show()

def main():
    """
    Pipeline principal d'entraînement.
    """
    # Parser d'arguments
    parser = argparse.ArgumentParser(description='Entraînement de modèles sur Mini-ImageNet')
    parser.add_argument('--epochs', type=int, default=5, help='Nombre d\'epochs (défaut: 5)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (défaut: 0.001)')
    parser.add_argument('--model', type=str, required=True, choices=['convnext', 'swin'],
                       help='Modèle à entraîner (convnext ou swin)')
    parser.add_argument('--exp_name', type=str, required=True,
                       help='Nom de l\'expérience (dossier de sauvegarde)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (défaut: 32)')
    args = parser.parse_args()
    
    # Créer le dossier de sauvegarde
    save_dir = Path("checkpoints") / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("  CONFIGURATION")
    print("="*60)
    print(f"  Modèle : {args.model}")
    print(f"  Expérience : {args.exp_name}")
    print(f"  Epochs : {args.epochs}")
    print(f"  Learning Rate : {args.lr}")
    print(f"  Batch Size : {args.batch_size}")
    print(f"  Dossier sauvegarde : {save_dir}")
    
    print("\n" + "="*60)
    print("  CHARGEMENT DES DONNÉES")
    print("="*60)
    
    train_loader, train_classes = get_train_loader(batch_size=args.batch_size)
    val_loader, val_classes = get_val_loader(batch_size=args.batch_size)
    
    print(f"\n   Classes d'entraînement : {train_classes}")
    print(f"   Classes de validation : {val_classes}")
    
    # Vérifier que les classes correspondent
    if set(train_classes) != set(val_classes):
        print("\n    ATTENTION : Les classes ne correspondent pas !")
        return
    
    # Instancier le modèle choisi
    if args.model == 'convnext':
        print("\n🔹 MODÈLE : ConvNeXt-Tiny")
        model = get_convnext_model(NUM_CLASSES)
        model_display_name = "ConvNeXt"
    else:  # swin
        print("\n🔹 MODÈLE : Swin Transformer-Tiny")
        model = get_swin_model(NUM_CLASSES)
        model_display_name = "Swin Transformer"
    
    # Entraîner le modèle
    history, best_val_acc = train_model(
        model, args.model, train_loader, val_loader, 
        args.epochs, args.lr, save_dir
    )
    plot_training_history(history, model_display_name, save_dir)
    
    # Résumé
    print("\n" + "="*60)
    print("  RÉSULTATS D'ENTRAÎNEMENT")
    print("="*60)
    print(f"\n  VALIDATION (Mini-ImageNet) :")
    print(f"    - {model_display_name}: {best_val_acc:.2f}%")
    print(f"\n  Checkpoints sauvegardés dans : {save_dir}")
    print(f"\n   Pour tester sur ImageNet-R, lancez: python3 test.py\n")

if __name__ == "__main__":
    main()
