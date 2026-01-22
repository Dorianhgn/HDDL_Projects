"""
Script de test pour ConvNeXt et Swin Transformer sur ImageNet-R.

Charge les meilleurs poids entraînés et teste sur ImageNet-R
pour mesurer la robustesse au changement de style.

Usage:
    python3 test.py                    # Teste tous les modèles
    python3 test.py --model convnext   # Teste seulement ConvNeXt
    python3 test.py --model swin       # Teste seulement Swin
"""

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import argparse
import json

from dataloader import get_test_loader
from models import get_convnext_model, get_swin_model

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 21
SAVE_DIR = Path("checkpoints")

def test_model(model, model_name, test_loader, device):
    """
    Teste le modèle sur ImageNet-R (mesure de robustesse).
    """
    model.eval()
    correct = 0
    total = 0
    
    # Statistiques par classe
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Test {model_name}"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images) # Prédictions
            _, predicted = outputs.max(1) # Classe prédite
            
            # Statistiques globales
            total += labels.size(0) # Nombre total d'images
            correct += predicted.eq(labels).sum().item() # 
            
            # Statistiques par classe
            for label, pred in zip(labels, predicted):
                label_item = label.item()
                if label_item not in class_correct:
                    class_correct[label_item] = 0
                    class_total[label_item] = 0
                class_total[label_item] += 1
                if pred == label:
                    class_correct[label_item] += 1
    
    test_acc = 100. * correct / total
    
    # Préparer les résultats détaillés par classe
    class_results = {}
    for class_idx in sorted(class_total.keys()):
        acc = 100. * class_correct[class_idx] / class_total[class_idx]
        class_results[class_idx] = {
            'correct': class_correct[class_idx],
            'total': class_total[class_idx],
            'accuracy': acc
        }
    
    # Afficher les résultats par classe
    print(f"\n   Détails par classe :")
    for class_idx in sorted(class_total.keys()):
        acc = class_results[class_idx]['accuracy']
        correct_count = class_results[class_idx]['correct']
        total_count = class_results[class_idx]['total']
        print(f"    Classe {class_idx}: {acc:.2f}% ({correct_count}/{total_count})")
    
    return test_acc, class_results

def load_and_test(model_name, model_fn, test_loader):
    """
    Charge le meilleur checkpoint et teste le modèle.
    
    Args:
        model_name: Nom du modèle ('convnext' ou 'swin')
        model_fn: Fonction pour créer le modèle
        test_loader: DataLoader de test
    """
    print(f"\n{'='*60}")
    print(f"  TEST : {model_name.upper()}")
    print(f"{'='*60}\n")
    print(f"  Device : {DEVICE}")
    print(f"  Dataset : ImageNet-R (images stylisées)")
    print(f"  Nombre de classes : {NUM_CLASSES}\n")
    
    # Créer le modèle
    model = model_fn(NUM_CLASSES).to(DEVICE)
    
    # Charger le meilleur checkpoint
    # Utiliser le dossier d'expérience spécifique pour swin
    exp_folder = "swin_lr0001_bs64" if model_name == "swin" else model_name
    checkpoint_path = SAVE_DIR / exp_folder / f"{model_name}_best.pth"
    
    if not checkpoint_path.exists():
        print(f" Checkpoint introuvable : {checkpoint_path}")
        print(f"   Entraîne d'abord le modèle avec: python3 train.py")
        return None
    
    print(f" Chargement du checkpoint : {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # Gérer les différents formats de checkpoint
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'N/A')
            val_acc = checkpoint.get('val_acc', 'N/A')
            print(f" Modèle chargé (Epoch {epoch}, Val Acc: {val_acc:.2f}%)")
        else:
            # Format ancien (seulement state_dict)
            model.load_state_dict(checkpoint)
            print(f" Modèle chargé")
    except Exception as e:
        print(f" Erreur lors du chargement : {e}")
        return None
    
    # Tester sur ImageNet-R
    print(f"\n Test sur ImageNet-R (robustesse)...")
    test_acc, class_results = test_model(model, model_name, test_loader, DEVICE)
    
    print(f"\n Test terminé !")
    print(f"  Précision sur ImageNet-R : {test_acc:.2f}%\n")
    
    return test_acc, class_results

def main():
    """
    Pipeline principal de test.
    """
    parser = argparse.ArgumentParser(description='Test des modèles sur ImageNet-R')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['all', 'convnext', 'swin'],
                       help='Modèle à tester (default: all)')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  CHARGEMENT DU DATASET DE TEST")
    print("="*60)
    
    test_loader, test_classes = get_test_loader()
    print(f"\n   Classes de test : {test_classes}")
    print(f"   Nombre d'images : {len(test_loader.dataset)}")
    
    results = {}
    detailed_results = {
        'class_names': test_classes,
        'models': {}
    }
    
    # Tester ConvNeXt
    if args.model in ['all', 'convnext']:
        convnext_result = load_and_test('convnext', get_convnext_model, test_loader)
        if convnext_result is not None:
            convnext_acc, convnext_class_results = convnext_result
            results['ConvNeXt'] = convnext_acc
            detailed_results['models']['convnext'] = {
                'global_accuracy': convnext_acc,
                'class_results': convnext_class_results
            }
    
    # Tester Swin Transformer
    if args.model in ['all', 'swin']:
        swin_result = load_and_test('swin', get_swin_model, test_loader)
        if swin_result is not None:
            swin_acc, swin_class_results = swin_result
            results['Swin Transformer'] = swin_acc
            detailed_results['models']['swin'] = {
                'global_accuracy': swin_acc,
                'class_results': swin_class_results
            }
    
    # Résumé final
    if results:
        print("\n" + "="*60)
        print("  RÉSULTATS FINAUX - ROBUSTESSE")
        print("="*60)
        print(f"\n  TEST (ImageNet-R images stylisées) :")
        for model_name, acc in results.items():
            print(f"    - {model_name:20s}: {acc:.2f}%")
        print(f"\n   Interprétation : Ces scores mesurent la robustesse")
        print(f"     des modèles face au changement de style (photos → dessins)\n")        
        # Sauvegarder les résultats dans un fichier JSON
        results_file = Path("checkpoints/res_test_convnext_vs_swin.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        print(f" Résultats détaillés sauvegardés : {results_file}")
        print(f"   → Chargez-les dans notebook4.ipynb avec: json.load(open('{results_file}'))\n")
if __name__ == "__main__":
    main()
 