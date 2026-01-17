"""
TASK 3: A -> B -> C
Trouve la Sphère Grise. 
Regarde quel est l'objet le plus PROCHE d'elle. 
Maintenant, quelle est la couleur de l'objet le plus LOIN de ce voisin ?
"""

import json
import os
from tqdm import tqdm
import numpy as np

# Mapping couleurs
COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
COLOR_TO_IDX = {c: i for i, c in enumerate(COLORS)}

def process_complex_task(json_path, split_name):
    print(f"Traitement Multi-Hop de {split_name}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    task_data = [] 
    
    # ANCHOR : Le point de départ (ex: Sphère Grise)
    START_SHAPE = 'sphere'
    START_COLOR = 'gray'

    for scene in tqdm(data['scenes']):
        img_name = scene['image_filename']
        objects = scene['objects']
        
        # 1. ÉTAPE A : Trouver le point de départ
        starts = [o for o in objects if o['shape'] == START_SHAPE and o['color'] == START_COLOR]
        
        # On a besoin d'au moins 3 objets pour faire une chaîne A -> B -> C
        if len(starts) == 1 and len(objects) >= 3:
            obj_a = starts[0]
            pos_a = np.array(obj_a['3d_coords'])
            
            # 2. ÉTAPE B : Trouver le plus PROCHE de A (Le Pivot)
            min_dist = float('inf')
            obj_b = None
            
            for o in objects:
                if o is obj_a: continue
                pos = np.array(o['3d_coords'])
                dist = np.linalg.norm(pos_a - pos)
                if dist < min_dist:
                    min_dist = dist
                    obj_b = o
            
            # 3. ÉTAPE C : Trouver le plus LOIN de B (La Cible)
            # C'est là que le CNN se perd : il doit calculer la distance depuis B, pas A !
            if obj_b:
                pos_b = np.array(obj_b['3d_coords'])
                max_dist_from_b = -1
                obj_c = None
                
                for o in objects:
                    if o is obj_b: continue # On peut retomber sur A, c'est autorisé, ou on l'exclut selon ta préférence
                    
                    pos = np.array(o['3d_coords'])
                    dist = np.linalg.norm(pos_b - pos)
                    if dist > max_dist_from_b:
                        max_dist_from_b = dist
                        obj_c = o
                
                # Le label est la couleur de l'objet C
                if obj_c:
                    label = COLOR_TO_IDX[obj_c['color']]
                    task_data.append({"image": img_name, "label": label})

    # Sauvegarde
    filename = f"task3_{split_name}.json"
    with open(filename, 'w') as f:
        json.dump(task_data, f)
        
    print(f"[{split_name}] Dataset Multi-Hop généré : {len(task_data)} images")

if __name__ == "__main__":
    # Adapte les chemins
    ROOT = "data_clevr/CLEVR_v1.0"
    process_complex_task(os.path.join(ROOT, "scenes", "CLEVR_train_scenes.json"), "train")
    process_complex_task(os.path.join(ROOT, "scenes", "CLEVR_val_scenes.json"), "val")