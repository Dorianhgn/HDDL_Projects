import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm 
import os
import numpy as np
import argparse
import yaml
import importlib


def parse_args():
    parser = argparse.ArgumentParser(description='Train UNet for segmentation')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    # parser.add_argument('--n_s', type=int, default=3, help='Number of scales')
    # parser.add_argument('--n_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW')
    parser.add_argument('--task', type=str, default='contours', help='Task type: classification, contours, segmentation, box, ...')
    parser.add_argument('--criterion', type=str, default='CrossEntropyLoss', help='Loss function: CrossEntropyLoss, BCEWithLogitsLoss, MSELoss')
    parser.add_argument('--model', type=str, default='models.unet.UNet', help='Dotted path to model class, e.g., models.unet.UNet')
    parser.add_argument('--model_args', type=dict, default=None, help='Model kwargs (when not using YAML)')
    parser.add_argument('--data_args', type=dict, default=None, help='Dataloader kwargs (when not using YAML)')
    parser.add_argument('--path', type=str, default='models/unet_ns3/', help='Path to save models')
    parser.add_argument('--continue_training', action='store_true', help='Continue from last checkpoint')
    parser.add_argument('--data_path', type=str, default='data/oxford-iiit-pet', help='Path to dataset')
    
    args = parser.parse_args()
    
    # Load config from YAML if provided
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Normalize model_args: allow dict or list-of-dicts
        if 'model_args' in config and isinstance(config['model_args'], list):
            merged = {}
            for item in config['model_args']:
                if isinstance(item, dict):
                    merged.update(item)
            config['model_args'] = merged
        
        # Normalize data_args: allow dict or list-of-dicts
        if 'data_args' in config and isinstance(config['data_args'], list):
            merged = {}
            for item in config['data_args']:
                if isinstance(item, dict):
                    merged.update(item)
            config['data_args'] = merged
        
        # Override defaults with config file values
        for key, value in config.items():
            setattr(args, key, value)
    
    return args


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    for images, targets in tqdm(loader, desc="Training"):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Handle BCEWithLogitsLoss shape mismatch (outputs: [N, 1], targets: [N])
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            loss = criterion(outputs.view(-1), targets.float())
        else:
            loss = criterion(outputs, targets)
            
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_val = 0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validation"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            
            # Handle BCEWithLogitsLoss shape mismatch (outputs: [N, 1], targets: [N])
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                loss = criterion(outputs.view(-1), targets.float())
            else:
                loss = criterion(outputs, targets)
                
            running_val += loss.item()
    
    return running_val / len(loader)


def resolve_model_class(dotted_path: str):
    module_name, class_name = dotted_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    # Try exact match first
    if hasattr(module, class_name):
        return getattr(module, class_name)
    # Fallbacks for simple capitalization differences
    candidates = [class_name.upper(), class_name.lower(), class_name.title()]
    for cand in candidates:
        if hasattr(module, cand):
            return getattr(module, cand)
    raise AttributeError(f"Classe '{class_name}' introuvable dans le module '{module_name}'.")


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
    
    # Save config to path
    config_save_path = os.path.join(args.path, "config.yaml")

    # Check if config exists, and ask before overwriting
    if os.path.exists(config_save_path):
        print(f"⚠️ Le fichier de configuration existe déjà à : {config_save_path}. Un modèle pourrait déjà y être entraîné.")
        response = input("Voulez-vous l'écraser ? (y/n) : ")
        if response not in ['o', 'oui', 'y', 'yes']:
            print("Abandon de la sauvegarde de la configuration. Fin du programme.")
            return
        
    with open(config_save_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f"📝 Configuration sauvegardée dans : {config_save_path}")
    
    # Load data
    from dataloader_segmentation import get_oxford_loaders
    
    # Prepare dataloader kwargs
    data_kwargs = {'task': args.task, 'batch_size': args.batch_size}
    if args.data_args is not None:
        # Merge data_args from YAML/CLI
        data_kwargs.update(args.data_args)
    
    loaders = get_oxford_loaders(args.data_path, **data_kwargs)
    
    # Initialize model (dynamic import from YAML/CLI)
    try:
        ModelClass = resolve_model_class(args.model)
    except Exception as e:
        raise RuntimeError(f"Impossible de charger le modèle '{args.model}': {e}")

    model_kwargs = {}
    if args.model_args is not None:
        # Already dict when using CLI; after YAML load it's normalized above
        model_kwargs = args.model_args
    # else:
    #     # Backward-compatibility defaults for UNet-like signatures
    #     model_kwargs = {'n_classes': args.n_classes, 'n_s': args.n_s}

    model = ModelClass(**model_kwargs).to(device)

    # Choose criterion (parameterizable)
    criterion_map = {
        'CrossEntropyLoss': nn.CrossEntropyLoss(),
        'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(),
        'MSELoss': nn.MSELoss(),
    }
    criterion_name = args.criterion if hasattr(args, 'criterion') else 'CrossEntropyLoss'
    if criterion_name not in criterion_map:
        raise ValueError(f"Critère '{criterion_name}' non supporté. Choisissez parmi: {list(criterion_map.keys())}")
    criterion = criterion_map[criterion_name]
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
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
        
        # Step scheduler
        scheduler.step()
    
    print("✅ Fin de l'entraînement !")


if __name__ == '__main__':
    main()
