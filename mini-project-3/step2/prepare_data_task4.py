"""
TASK 4: A --material--> B --shape--> C
    Trouve la Sphère Grise. 
    Regarde quel est son MATÉRIAU (ex: Métal). 
    Trouve le CYLINDRE fait de ce MÊME MATÉRIAU. 
    Quelle est la couleur de l'objet le plus PETIT à côté de ce cylindre ?
"""

import json
import os
from tqdm import tqdm
import numpy as np

# Mapping couleurs
COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
COLOR_TO_IDX = {c: i for i, c in enumerate(COLORS)}

def process_attribute_relay(json_path, split_name):
    print(f"Traitement Attribute Relay de {split_name}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    task_data = []
    task_metadata = [] 
    
    # 1. L'ANCRE (Départ fixe pour aider le modèle à démarrer)
    START_SHAPE = 'sphere'
    START_COLOR = 'gray'
    
    # 2. LA CIBLE INTERMÉDIAIRE (On cherchera cette forme avec le MEME MATERIAU)
    RELAY_SHAPE = 'cylinder'

    for scene in tqdm(data['scenes']):
        img_name = scene['image_filename']
        objects = scene['objects']
        
        # --- ÉTAPE A : Trouver l'Ancre (Sphère Grise) ---
        anchors = [o for o in objects if o['shape'] == START_SHAPE and o['color'] == START_COLOR]
        
        if len(anchors) == 1:
            obj_a = anchors[0]
            target_material = obj_a['material'] # ex: 'metal' ou 'rubber'
            
            # --- ÉTAPE B : Trouver le Relais (Cylindre fait du MÊME MATÉRIAU) ---
            # On cherche tous les cylindres qui ont le matériau de A
            candidates_b = [o for o in objects if o['shape'] == RELAY_SHAPE and o['material'] == target_material]
            
            # S'il y en a plusieurs, on prend le plus proche de A pour lever l'ambiguïté
            if candidates_b:
                pos_a = np.array(obj_a['3d_coords'])
                # Tri par distance à A
                obj_b = min(candidates_b, key=lambda o: np.linalg.norm(pos_a - np.array(o['3d_coords'])))
                
                # --- ÉTAPE C : Trouver la Cible Finale (Objet le plus proche de B, sauf B et A) ---
                pos_b = np.array(obj_b['3d_coords'])
                
                min_dist = float('inf')
                obj_c = None
                
                for o in objects:
                    if o is obj_b or o is obj_a: continue 
                    
                    pos = np.array(o['3d_coords'])
                    dist = np.linalg.norm(pos_b - pos)
                    
                    if dist < min_dist:
                        min_dist = dist
                        obj_c = o
                
                # Le Label est la couleur de cet objet C
                if obj_c:
                    label = COLOR_TO_IDX[obj_c['color']]
                    task_data.append({"image": img_name, "label": label})
                    
                    # Extraire les métadonnées (A = ancre, B = relais, C = cible)
                    anchor_meta = {
                        'color': obj_a['color'],
                        'material': obj_a['material'],
                        'shape': obj_a['shape'],
                        'size': obj_a['size'],
                        'pixel_coords': obj_a['pixel_coords'],
                        '3d_coords': obj_a['3d_coords']
                    }
                    relay_meta = {
                        'color': obj_b['color'],
                        'material': obj_b['material'],
                        'shape': obj_b['shape'],
                        'size': obj_b['size'],
                        'pixel_coords': obj_b['pixel_coords'],
                        '3d_coords': obj_b['3d_coords']
                    }
                    target_meta = {
                        'color': obj_c['color'],
                        'material': obj_c['material'],
                        'shape': obj_c['shape'],
                        'size': obj_c['size'],
                        'pixel_coords': obj_c['pixel_coords'],
                        '3d_coords': obj_c['3d_coords']
                    }
                    task_metadata.append({
                        "image": img_name,
                        "anchor": anchor_meta,
                        "relay": relay_meta,
                        "target": target_meta
                    })

    # Sauvegarde
    filename = f"task4_{split_name}.json"
    with open(filename, 'w') as f:
        json.dump(task_data, f)
    
    metadata_filename = f"task4_{split_name}_metadata.json"
    with open(metadata_filename, 'w') as f:
        json.dump(task_metadata, f)
        
    print(f"[{split_name}] Dataset Relay généré : {len(task_data)} images")
    print(f"[{split_name}] Metadata généré : {len(task_metadata)} entrées")

if __name__ == "__main__":
    ROOT = "data_clevr/CLEVR_v1.0"
    process_attribute_relay(os.path.join(ROOT, "scenes", "CLEVR_train_scenes.json"), "train")
    process_attribute_relay(os.path.join(ROOT, "scenes", "CLEVR_val_scenes.json"), "val")