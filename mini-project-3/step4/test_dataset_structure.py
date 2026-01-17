"""
Script pour inspecter la structure du dataset timm/mini-imagenet
"""
from datasets import load_dataset

print("🔍 Inspection du dataset timm/mini-imagenet...\n")

try:
    dataset = load_dataset('timm/mini-imagenet', split='train', streaming=True, trust_remote_code=True)
    
    print("✅ Dataset chargé !")
    print("\n📋 Inspection des 5 premiers échantillons :\n")
    
    for i, sample in enumerate(dataset):
        if i >= 5:
            break
        
        print(f"Échantillon {i+1}:")
        print(f"  Clés disponibles: {list(sample.keys())}")
        
        for key, value in sample.items():
            if key != 'image':
                print(f"    {key}: {value} (type: {type(value).__name__})")
        
        print()
        
except Exception as e:
    print(f"❌ Erreur: {e}")
