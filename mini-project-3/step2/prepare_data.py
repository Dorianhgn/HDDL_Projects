import json
import os
from tqdm import tqdm # pip install tqdm si pas installé

# Chemins (lancer sous le répertoire mini-project-3/step3 : `python -m prepare_data`)
ROOT_DIR = "data_clevr/CLEVR_v1.0"
SCENES_PATH_TRAIN = os.path.join(ROOT_DIR, "scenes", "CLEVR_train_scenes.json")
SCENES_PATH_VAL = os.path.join(ROOT_DIR, "scenes", "CLEVR_val_scenes.json")

# Mapping couleurs
COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
COLOR_TO_IDX = {c: i for i, c in enumerate(COLORS)}

def process_scenes(json_path, split_name):
    print(f"Traitement de {split_name}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    task1_data = [] # (filename, label)
    task2_data = [] # (filename, label)

    for scene in tqdm(data['scenes']):
        img_name = scene['image_filename']
        objects = scene['objects']
        
        # --- TACHE 1 : Cylindre le plus loin ---
        # CORRECTION : On utilise pixel_coords[2] qui est la profondeur (distance caméra)
        cylinders = [o for o in objects if o['shape'] == 'cylinder']
        
        if cylinders:
            # On cherche celui qui a la plus grande profondeur (le plus loin)
            furthest = max(cylinders, key=lambda x: x['pixel_coords'][2])
            label = COLOR_TO_IDX[furthest['color']]
            task1_data.append({"image": img_name, "label": label})

        # --- TACHE 2 : Voisin de la Sphère Grise ---
        # Ici on garde 3d_coords car on veut la proximité spatiale réelle
        gray_spheres = [o for o in objects if o['shape'] == 'sphere' and o['color'] == 'gray']
        
        if len(gray_spheres) == 1 and len(objects) > 1:
            anchor = gray_spheres[0]
            anchor_pos = anchor['3d_coords'] # [x, y, z]
            
            # On cherche l'objet le plus proche/lointain (distance Euclidienne 3D)
            # On exclut l'ancre elle-même
            dists = []
            for obj in objects:
                if obj is anchor: continue
                
                # Calcul distance 3D : sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)
                obj_pos = obj['3d_coords']
                d = sum((c1 - c2)**2 for c1, c2 in zip(anchor_pos, obj_pos))**0.5
                dists.append((d, obj))
            
            if dists:
                # On prend le min (le plus proche) ou le max (le plus lointain) selon la tâche
                closest_obj = max(dists, key=lambda x: x[0])[1]
                label = COLOR_TO_IDX[closest_obj['color']]
                task2_data.append({"image": img_name, "label": label})

    # Sauvegarde
    with open(f"task1_{split_name}.json", 'w') as f:
        json.dump(task1_data, f)
    with open(f"task2_{split_name}.json", 'w') as f:
        json.dump(task2_data, f)
        
    print(f"[{split_name}] Tâche 1 (Cylindre Loin): {len(task1_data)} images")
    print(f"[{split_name}] Tâche 2 (Voisin Sphère): {len(task2_data)} images")

if __name__ == "__main__":
    process_scenes(SCENES_PATH_TRAIN, "train")
    process_scenes(SCENES_PATH_VAL, "val")