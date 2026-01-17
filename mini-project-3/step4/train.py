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

from dataloader import get_train_loader, get_val_loader
from models import get_convnext_model, get_swin_model

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 21
EPOCHS = 5
LEARNING_RATE = 0.001
SAVE_DIR = Path("checkpoints")
SAVE_DIR.mkdir(exist_ok=True)

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Entraîne le modèle pour une epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
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

def train_model(model, model_name, train_loader, val_loader, epochs=EPOCHS):
    """
    Pipeline d'entraînement complet.
    """
    print(f"\n{'='*60}")
    print(f"  ENTRAÎNEMENT : {model_name}")
    print(f"{'='*60}\n")
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {epochs}")
    print(f"  Learning Rate : {LEARNING_RATE}")
    print(f"  Nombre de classes : {NUM_CLASSES}\n")
    
    model = model.to(DEVICE)
    
    # Loss et optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
    SAVE_DIR_PATH = SAVE_DIR / model_name
    SAVE_DIR_PATH.mkdir(exist_ok=True)
    
    # Boucle d'entraînement
    for epoch in range(epochs):
        # Entraînement
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch
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

def plot_training_history(history, model_name):
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
    plot_path = SAVE_DIR / f"{model_name}_training.png"
    plt.savefig(plot_path, dpi=150)
    print(f"   Courbes sauvegardées : {plot_path}")
    plt.show()

def main():
    """
    Pipeline principal d'entraînement.
    """
    print("\n" + "="*60)
    print("  CHARGEMENT DES DONNÉES")
    print("="*60)
    
    train_loader, train_classes = get_train_loader()
    val_loader, val_classes = get_val_loader()
    
    print(f"\n   Classes d'entraînement : {train_classes}")
    print(f"   Classes de validation : {val_classes}")
    
    # Vérifier que les classes correspondent
    if set(train_classes) != set(val_classes):
        print("\n    ATTENTION : Les classes ne correspondent pas !")
        return
    
    # Entraîner ConvNeXt
    print("\n🔹 MODÈLE 1 : ConvNeXt-Tiny")
    convnext_model = get_convnext_model(NUM_CLASSES)
    convnext_history, convnext_val = train_model(
        convnext_model, "convnext", train_loader, val_loader
    )
    plot_training_history(convnext_history, "ConvNeXt")
    
    # Entraîner Swin Transformer
    print("\n🔹 MODÈLE 2 : Swin Transformer-Tiny")
    swin_model = get_swin_model(NUM_CLASSES)
    swin_history, swin_val = train_model(
        swin_model, "swin", train_loader, val_loader
    )
    plot_training_history(swin_history, "Swin Transformer")
    
    # Résumé
    print("\n" + "="*60)
    print("  RÉSULTATS D'ENTRAÎNEMENT")
    print("="*60)
    print(f"\n  VALIDATION (Mini-ImageNet) :")
    print(f"    - ConvNeXt :        {convnext_val:.2f}%")
    print(f"    - Swin Transformer: {swin_val:.2f}%")
    print(f"\n   Pour tester sur ImageNet-R, lancez: python3 test.py\n")

if __name__ == "__main__":
    main()
