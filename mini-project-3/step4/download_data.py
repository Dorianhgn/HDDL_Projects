"""
Script pour télécharger les 3 datasets nécessaires au projet :
1. Mini-ImageNet Train (photos réelles) - TOUTES les images
2. Mini-ImageNet Validation (photos réelles) - TOUTES les images
3. ImageNet-R Test (images stylisées) - TOUTES les images
"""

import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Configuration - chemins relatifs au script
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
TRAIN_OUTPUT_DIR = DATA_DIR / "mini-imagenet-train"
VAL_OUTPUT_DIR = DATA_DIR / "mini-imagenet-val"
TEST_OUTPUT_DIR = DATA_DIR / "imagenet-r-mini"
MAX_IMAGES_PER_CLASS = None  # None = prendre TOUTES les images disponibles

# Mapping des 21 classes communes entre Mini-ImageNet et ImageNet-R
# Pour Mini-ImageNet : {label_numérique: nom_classe}
MINI_IMAGENET_LABELS = {
    5: "toucan",
    6: "goose",
    7: "jellyfish",
    14: "golden_retriever",
    17: "boxer",
    19: "french_bulldog",
    21: "dalmatian",
    27: "lion",
    28: "meerkat",
    29: "ladybug",
    31: "ant",
    40: "cannon",
    41: "carousel",
    52: "electric_guitar",
    62: "lipstick",
    64: "missile",
    74: "school_bus",
    79: "spider_web",
    81: "tank",
    87: "vase",
    94: "hotdog"
}

# Pour ImageNet-R : {wnid: nom_classe}
IMAGENET_R_WNIDS = {
    "n01843383": "toucan",
    "n01855672": "goose",
    "n01910747": "jellyfish",
    "n02099601": "golden_retriever",
    "n02108089": "boxer",
    "n02108915": "french_bulldog",
    "n02110341": "dalmatian",
    "n02129165": "lion",
    "n02138441": "meerkat",
    "n02165456": "ladybug",
    "n02219486": "ant",
    "n02950826": "cannon",
    "n02966193": "carousel",
    "n03272010": "electric_guitar",
    "n03676483": "lipstick",
    "n03773504": "missile",
    "n04146614": "school_bus",
    "n04275548": "spider_web",
    "n04389033": "tank",
    "n04522168": "vase",
    "n07697537": "hotdog"
}

def download_imagenet_subset(split='train', output_dir=TRAIN_OUTPUT_DIR):
    """
    Télécharge un sous-ensemble de Mini-ImageNet avec les 21 classes cibles.
    Utilise les labels numériques pour filtrer les classes.
    Télécharge TOUTES les images disponibles.
    
    Args:
        split: 'train' ou 'validation'
        output_dir: dossier de sortie
    """
    max_per_class = "TOUTES" if MAX_IMAGES_PER_CLASS is None else MAX_IMAGES_PER_CLASS
    print(f"📥 Téléchargement de Mini-ImageNet - Split: {split.upper()}")
    print(f"📂 Destination : {output_dir}")
    print(f"🎯 Classes : {list(MINI_IMAGENET_LABELS.values())}")
    print(f"📊 Images par classe : {max_per_class}\n")
    
    # Créer le dossier de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compteurs
    counters = {name: 0 for name in MINI_IMAGENET_LABELS.values()}
    
    try:
        # Charger Mini-ImageNet depuis timm
        print("⏳ Chargement du dataset Mini-ImageNet...")
        dataset = load_dataset('timm/mini-imagenet', split=split, streaming=True, trust_remote_code=True)
        
        print("✅ Dataset chargé ! Début du filtrage...\n")
        
        # Parcourir le dataset
        for i, sample in enumerate(tqdm(dataset, desc=f"Téléchargement {split}")):
            # Récupérer le label numérique de l'échantillon
            sample_label = sample.get('label', None)
            
            if sample_label in MINI_IMAGENET_LABELS:
                class_name = MINI_IMAGENET_LABELS[sample_label]
                
                # Si MAX_IMAGES_PER_CLASS est None, pas de limite
                should_save = (MAX_IMAGES_PER_CLASS is None) or (counters[class_name] < MAX_IMAGES_PER_CLASS)
                
                if should_save:
                    # Créer le dossier de classe
                    class_dir = output_dir / class_name
                    class_dir.mkdir(exist_ok=True)
                    
                    # Sauvegarder l'image
                    file_path = class_dir / f"{class_name}_{counters[class_name]:04d}.jpg"
                    sample['image'].convert("RGB").save(file_path)
                    
                    counters[class_name] += 1
                    
                    # Afficher le progrès
                    if counters[class_name] % 50 == 0:
                        print(f"  ✓ {class_name}: {counters[class_name]} images")
            
            # Arrêter si toutes les classes sont complètes (seulement si limite définie)
            if MAX_IMAGES_PER_CLASS is not None and all(c >= MAX_IMAGES_PER_CLASS for c in counters.values()):
                print("\n✅ Toutes les classes sont complètes !")
                break
            
            # Sécurité : arrêter après avoir scanné beaucoup d'images
            if i > 200000:
                print("\n⚠️  Arrêt de sécurité après 200k images scannées.")
                break
        
    except Exception as e:
        print(f"\n Erreur lors du téléchargement : {e}")
        print("\n💡 Alternative : Vérifie que le dataset 'timm/mini-imagenet' est accessible")
        print("   ou télécharge manuellement depuis https://github.com/huggingface/pytorch-image-models")
        return False
    
    # Afficher le résumé
    print("\n" + "="*50)
    print(f" RÉSUMÉ DU TÉLÉCHARGEMENT - {split.upper()}")
    print("="*50)
    total = 0
    for class_name, count in counters.items():
        print(f"  {class_name:15s} : {count:4d} images")
        total += count
    print(f"\n  TOTAL : {total} images\n")
    
    return True

def download_imagenet_r_test():
    """
    Télécharge ImageNet-R (images stylisées) pour le test de robustesse.
    Utilise les WNIDs pour filtrer les 21 classes cibles.
    Télécharge TOUTES les images disponibles pour chaque classe.
    """
    output_dir = TEST_OUTPUT_DIR
    max_images = None  # None = prendre TOUTES les images
    
    max_per_class = "TOUTES" if max_images is None else max_images
    print(f"📥 Téléchargement de ImageNet-R - TEST (images stylisées)")
    print(f"📂 Destination : {output_dir}")
    print(f"🎯 Classes : {list(IMAGENET_R_WNIDS.values())}")
    print(f"📊 Images par classe : {max_per_class}\n")
    
    # Créer le dossier de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compteurs
    counters = {name: 0 for name in IMAGENET_R_WNIDS.values()}
    
    try:
        # Charger ImageNet-R
        print("⏳ Chargement du dataset ImageNet-R...")
        dataset = load_dataset('axiong/imagenet-r', split='test', streaming=True)
        
        print("✅ Dataset chargé ! Début du filtrage...\n")
        
        # Parcourir le dataset
        for i, sample in enumerate(tqdm(dataset, desc="Téléchargement ImageNet-R")):
            sample_wnid = sample.get('wnid', None)
            
            if sample_wnid in IMAGENET_R_WNIDS:
                class_name = IMAGENET_R_WNIDS[sample_wnid]
                
                # Si max_images est None, pas de limite
                should_save = (max_images is None) or (counters[class_name] < max_images)
                
                if should_save:
                    # Créer le dossier de classe
                    class_dir = output_dir / class_name
                    class_dir.mkdir(exist_ok=True)
                    
                    # Sauvegarder l'image
                    file_path = class_dir / f"{class_name}_{counters[class_name]:04d}.jpg"
                    sample['image'].convert("RGB").save(file_path)
                    
                    counters[class_name] += 1
                    
                    if counters[class_name] % 50 == 0:
                        print(f"  ✓ {class_name}: {counters[class_name]} images")
            
            # Arrêter si toutes les classes sont complètes (seulement si limite définie)
            if max_images is not None and all(c >= max_images for c in counters.values()):
                print("\n✅ Toutes les classes sont complètes !")
                break
            
            # Sécurité : arrêter après avoir scanné beaucoup d'images
            if i > 100000:
                print("\n  Arrêt de sécurité après 100k images scannées.")
                break
        
    except Exception as e:
        print(f"\n Erreur lors du téléchargement : {e}")
        return False
    
    # Afficher le résumé
    print("\n" + "="*50)
    print(" RÉSUMÉ DU TÉLÉCHARGEMENT - ImageNet-R")
    print("="*50)
    total = 0
    for class_name, count in counters.items():
        print(f"  {class_name:15s} : {count:4d} images")
        total += count
    print(f"\n  TOTAL : {total} images\n")
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("  TÉLÉCHARGEMENT COMPLET DES DATASETS")
    print("  Train + Validation + Test - TOUTES LES IMAGES")
    print("="*60 + "\n")
    
    # Télécharger le split d'entraînement
    print("\n SPLIT 1/3 : TRAIN (Mini-ImageNet - Photos réelles)\n")
    success_train = download_imagenet_subset(split='train', output_dir=TRAIN_OUTPUT_DIR)
    
    # Télécharger le split de validation
    print("\n SPLIT 2/3 : VALIDATION (Mini-ImageNet - Photos réelles)\n")
    success_val = download_imagenet_subset(split='validation', output_dir=VAL_OUTPUT_DIR)
    
    # Télécharger ImageNet-R pour le test
    print("\n SPLIT 3/3 : TEST (ImageNet-R - Images stylisées)\n")
    success_test = download_imagenet_r_test()
    
    success = success_train and success_val and success_test
    
    if not success:
        print("\n  Le téléchargement automatique a échoué.")
        print("\n OPTIONS ALTERNATIVES :")
        print("  1. Utiliser ImageNette (10 classes) : https://github.com/fastai/imagenette")
        print("  2. Télécharger manuellement depuis ImageNet : https://www.image-net.org/")
        print("  3. Utiliser un dataset miroir de Kaggle")
        print("\n Place les images dans :")
        print(f"   - Train: step4/{TRAIN_OUTPUT_DIR}/")
        print(f"   - Val:   step4/{VAL_OUTPUT_DIR}/")
        print(f"   - Test:  step4/{TEST_OUTPUT_DIR}/")
        print("   avec la structure : class_name/image_xxxx.jpg")
    else:
        print("\n" + "="*60)
        print("  ✅ TÉLÉCHARGEMENT COMPLET TERMINÉ")
        print("="*60)
        print("\n   Datasets prêts pour l'entraînement !")
        print(f"   Train: {TRAIN_OUTPUT_DIR}")
        print(f"   Val:   {VAL_OUTPUT_DIR}")
        print(f"   Test:  {TEST_OUTPUT_DIR}\n")
