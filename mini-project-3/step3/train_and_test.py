import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import Config


# ============================================================================
# Entraînement (Fine-tuning)
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Entraîne le modèle pour une époque"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Métriques
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Évalue le modèle sur le set de validation"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def train_model(model, model_name, train_loader, val_loader, num_epochs=5):
    """
    Entraîne un modèle et sauvegarde les métriques.
    
    Returns:
        history: Dict contenant les losses et accuracies
    """
    print("\n" + "="*70)
    print(f"🚀 Entraînement de {model_name}")
    print("="*70)
    
    model = model.to(Config.DEVICE)
    
    # Optimiseur (AdamW pour le fine-tuning)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Historique
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Entraînement
    for epoch in range(num_epochs):
        print(f"\n📍 Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)
        
        # Sauvegarde l'historique
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Affiche les résultats
        print(f"\n✅ Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Sauvegarde le modèle
    save_path = Config.RESULTS_DIR / f"{model_name}_final.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n💾 Modèle sauvegardé: {save_path}")
    
    return history