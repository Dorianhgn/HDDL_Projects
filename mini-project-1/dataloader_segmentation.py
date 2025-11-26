import os
import re
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as F
import random

class OxfordPetDataset(Dataset):
    def __init__(self, root_dir, split='train', task='contours', target_size=(256, 256), 
                 dataset_variant='custom', aug_percent=0.0, aug_rotation_limit=30, 
                 blur_radius=2.0, noise_level=0.1, sensitivity_type=None, train_test_split_ratio=0.8):
        """
        Classe Dataset principale pour le projet avec Data Augmentation Additive.
        
        Args:
            root_dir (str): Chemin vers le dossier 'data/oxford-iiit-pet'
            split (str): 'train', 'val' ou 'test'
            task (str): 'contours' (Segmentation masque) ou 'boxes' (Detection rectangle)
            target_size (tuple): Taille cible des images (H, W) ex: (256, 256)
            dataset_variant (str): 'original' (split officiel) ou 'custom' (split aléatoire complet)
            aug_percent (float): Pourcentage de copies augmentées à ajouter (0.0 = pas d'augmentation, 1.0 = doubler)
            aug_rotation_limit (int): Angle max de rotation (en degrés) pour l'augmentation random
            blur_radius (float): Rayon max du flou gaussien
            noise_level (float): Niveau max de bruit gaussien (écart-type)
            sensitivity_type (str): Type de transformation forcée pour test de sensibilité 
                                   ('flip', 'rotation', 'blur', 'noise', None)
            train_test_split_ratio (float): Ratio de split train/test (défaut: 0.8 = 80% train, 20% val+test)
        """
        self.root_dir = root_dir
        self.split = split
        self.task = task
        self.target_size = target_size
        self.dataset_variant = dataset_variant
        self.aug_percent = aug_percent if split == 'train' else 0.0
        self.aug_rotation_limit = aug_rotation_limit
        self.blur_radius = blur_radius
        self.noise_level = noise_level
        self.sensitivity_type = sensitivity_type if split == 'test' else None
        self.train_test_split_ratio = train_test_split_ratio
        
        # Chemins des sous-dossiers
        self.images_dir = os.path.join(root_dir, 'images')
        self.annotations_dir = os.path.join(root_dir, 'annotations')
        self.trimaps_dir = os.path.join(self.annotations_dir, 'trimaps')
        self.xmls_dir = os.path.join(self.annotations_dir, 'xmls')
        
        # 1. Initialisation de la liste des fichiers (DataFrame)
        self.df = self._load_and_split_data()
        
        # 2. Filtrage si on fait la tâche 'boxes' (car certains XML manquent)
        if self.task in ['boxes', 'box']:
            self._filter_missing_xmls()
        
        # 3. Calcul du nombre d'échantillons originaux et augmentés
        self.original_len = len(self.df)
        self.augmented_len = int(self.original_len * self.aug_percent)
        self.total_len = self.original_len + self.augmented_len
        
        if self.split == 'train' and self.aug_percent > 0:
            print(f"[{self.split}] Data Augmentation: {self.original_len} originaux + {self.augmented_len} augmentés = {self.total_len} total")
        
        if self.split == 'test' and self.sensitivity_type:
            print(f"[{self.split}] Test de sensibilité: transformation '{self.sensitivity_type}' appliquée à tout le dataset")

    def _parse_list_file(self, file_path):
        """
        Lit le fichier list.txt du dataset Oxford-IIIT Pet
        Format des lignes : Image_Name CLASS-ID SPECIES BREED-ID
        """
        data = []
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"Fichier introuvable : {file_path}")

        with open(file_path, 'r') as f:
            for line in f:
                # Ignorer les commentaires qui commencent par #
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split()
                if len(parts) != 4:
                    continue
                
                image_name = parts[0]
                class_id = int(parts[1])
                species_id = int(parts[2])
                breed_id = int(parts[3])
                
                # Extraire le nom de la race (tout avant le dernier underscore + chiffre)
                match = re.match(r'(.+)_\d+$', image_name)
                if match:
                    breed_name = match.group(1)
                else:
                    breed_name = image_name
                
                # Nom de l'espèce
                species_name = 'Cat' if species_id == 1 else 'Dog'
                
                # Catégorie pour cohérence avec le code précédent (0=Chat, 1=Chien)
                category = 0 if species_id == 1 else 1
                
                data.append({
                    'filename': image_name + '.jpg',
                    'breed': breed_name,
                    'class_id': class_id,
                    'species_id': species_id,
                    'species_name': species_name,
                    'category': str(category),
                    'breed_id': breed_id
                })
        
        return pd.DataFrame(data)

    def _load_and_split_data(self):
        """Méthode interne pour lire les fichiers .txt et séparer train/val"""
        
        if self.dataset_variant == 'original':
            trainval_path = os.path.join(self.annotations_dir, 'trainval.txt')
            test_path = os.path.join(self.annotations_dir, 'test.txt')
            
            def parse_txt(path):
                data = []
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Fichier introuvable : {path}")
                with open(path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            data.append(parts[0] + '.jpg')
                return pd.DataFrame(data, columns=['filename'])

            # Chargement selon le split demandé
            if self.split == 'test':
                return parse_txt(test_path)
            else:
                # On charge TOUT le trainval puis on coupe
                full_train_df = parse_txt(trainval_path)
                val_size = 1.0 - self.train_test_split_ratio
                train_df, val_df = train_test_split(full_train_df, test_size=val_size, random_state=42)
                
                if self.split == 'train':
                    return train_df.reset_index(drop=True)
                elif self.split == 'val':
                    return val_df.reset_index(drop=True)
        
        elif self.dataset_variant == 'custom':
            list_path = os.path.join(self.annotations_dir, 'list.txt')
            full_df = self._parse_list_file(list_path)
            
            # Split selon train_test_split_ratio (ex: 80% Train, 20% Val+Test)
            temp_size = 1.0 - self.train_test_split_ratio
            train_df, temp_df = train_test_split(full_df, test_size=temp_size, random_state=42)
            # Split 50% Val, 50% Test du reste
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
            
            if self.split == 'train':
                return train_df.reset_index(drop=True)
            elif self.split == 'val':
                return val_df.reset_index(drop=True)
            elif self.split == 'test':
                return test_df.reset_index(drop=True)
        else:
            raise ValueError(f"dataset_variant inconnu : {self.dataset_variant}")

    def _filter_missing_xmls(self):
        """Supprime les images qui n'ont pas de XML correspondant"""
        initial_len = len(self.df)
        self.df['xml_path'] = self.df['filename'].apply(
            lambda x: os.path.join(self.xmls_dir, x.replace('.jpg', '.xml'))
        )
        # On ne garde que si le fichier existe
        self.df = self.df[self.df['xml_path'].apply(os.path.exists)].copy()
        print(f"[{self.split}] Mode Boxes : {initial_len - len(self.df)} images ignorées (XML manquant).")

    def _resize_with_padding(self, img, is_mask=False):
        """Redimensionne en ajoutant des bandes noires pour ne pas déformer"""
        target_w, target_h = self.target_size
        w, h = img.size
        
        # Calcul du ratio
        ratio = min(target_w/w, target_h/h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        # Resize
        method = Image.NEAREST if is_mask else Image.BILINEAR
        img = img.resize((new_w, new_h), method)
        
        # Création fond noir
        new_img = Image.new(img.mode, (target_w, target_h), 0)
        # Collage au centre
        new_img.paste(img, ((target_w - new_w)//2, (target_h - new_h)//2))
        
        return new_img

    def _get_box_mask(self, filename, img_size):
        """Génère un masque binaire à partir du XML"""
        xml_path = os.path.join(self.xmls_dir, filename.replace('.jpg', '.xml'))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        mask = Image.new('L', img_size, 0)
        
        try:
            bndbox = root.find('object').find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            # Dessin du rectangle blanc
            from PIL import ImageDraw
            draw = ImageDraw.Draw(mask)
            draw.rectangle([xmin, ymin, xmax, ymax], fill=1)
        except:
            # Si XML mal formé, on renvoie un masque vide (sécurité)
            pass
            
        return mask

    def __len__(self):
        return self.total_len
    
    def _transform_image_mask(self, image, mask, transform_type=None):
        """
        Applique une transformation aléatoire (ou forcée) à l'image et au masque.
        
        Args:
            image (PIL.Image): Image en RGB
            mask (PIL.Image): Masque de segmentation
            transform_type (str): Type de transformation forcée (None = random parmi les 4)
        
        Returns:
            tuple: (image_transformée, masque_transformé)
        """
        # Si aucun type n'est spécifié, on choisit aléatoirement
        if transform_type is None:
            transform_type = random.choice(['flip', 'rotation', 'blur', 'noise'])
        
        if transform_type == 'flip':
            # Flip Horizontal - appliqué à l'image ET au masque
            image = F.hflip(image)
            mask = F.hflip(mask)
        
        elif transform_type == 'rotation':
            # Rotation aléatoire - appliquée à l'image ET au masque
            angle = random.uniform(-self.aug_rotation_limit, self.aug_rotation_limit)
            image = F.rotate(image, angle, interpolation=F.InterpolationMode.BILINEAR, fill=0)
            mask = F.rotate(mask, angle, interpolation=F.InterpolationMode.NEAREST, fill=0)
        
        elif transform_type == 'blur':
            # Flou Gaussien - appliqué UNIQUEMENT à l'image
            radius = random.uniform(0.5, self.blur_radius)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
            # Le masque reste inchangé
        
        elif transform_type == 'noise':
            # Bruit Gaussien - appliqué UNIQUEMENT à l'image
            # On doit d'abord convertir en array, ajouter le bruit, puis reconvertir
            img_array = np.array(image).astype(np.float32) / 255.0
            noise = np.random.normal(0, random.uniform(0, self.noise_level), img_array.shape)
            img_array = np.clip(img_array + noise, 0, 1)
            image = Image.fromarray((img_array * 255).astype(np.uint8))
            # Le masque reste inchangé
        
        return image, mask

    def __getitem__(self, idx):
        # Déterminer si c'est un échantillon original ou augmenté
        is_augmented = idx >= self.original_len
        
        if is_augmented:
            # Pour les copies augmentées, on revient à l'index original
            actual_idx = idx - self.original_len
        else:
            actual_idx = idx
        
        # 1. Récupérer le nom du fichier et les métadonnées
        row = self.df.iloc[actual_idx]
        img_name = row['filename']
        img_path = os.path.join(self.images_dir, img_name)
        
        # 2. Charger l'image
        image = Image.open(img_path).convert('RGB')
        
        # 3. Charger le Masque (Target) selon la tâche
        if self.task == 'contours':
            mask_name = img_name.replace('.jpg', '.png')
            mask_path = os.path.join(self.trimaps_dir, mask_name)
            mask = Image.open(mask_path) # Valeurs 1, 2, 3
        elif self.task in ['boxes', 'box']:
            mask = self._get_box_mask(img_name, image.size) # Valeurs 0, 1
        else:
            raise ValueError(f"Tâche inconnue : {self.task}")

        # 4. Appliquer la transformation si nécessaire
        if is_augmented:
            # Mode Train: transformation aléatoire pour les copies augmentées
            image, mask = self._transform_image_mask(image, mask, transform_type=None)
        elif self.sensitivity_type:
            # Mode Test: transformation forcée pour tout le dataset
            image, mask = self._transform_image_mask(image, mask, transform_type=self.sensitivity_type)

        # 5. Transformation (Resize + Padding)
        image = self._resize_with_padding(image, is_mask=False)
        mask = self._resize_with_padding(mask, is_mask=True)

        # 6. Conversion en Tenseurs PyTorch
        img_tensor = F.to_tensor(image) # Devient [0, 1] float
        
        mask_array = np.array(mask)
        mask_tensor = torch.from_numpy(mask_array).long() # Devient Entier
        
        # Correction des valeurs pour trimaps (1,2,3 -> 0,1,2)
        # S'applique pour 'contours', 'classification', et toute tâche utilisant trimaps
        if self.task not in ['boxes', 'box']:
            # Trimap : 1=Animal, 2=Fond, 3=Bord
            # Cible : 0=Fond, 1=Animal, 2=Bord
            # Le padding (0) reste 0 (Fond)
            
            # Mapping manuel pour être sûr
            new_mask = torch.zeros_like(mask_tensor)
            new_mask[mask_tensor == 1] = 1 # Animal
            new_mask[mask_tensor == 2] = 0 # Fond
            new_mask[mask_tensor == 3] = 2 # Bord
            mask_tensor = new_mask

        return img_tensor, mask_tensor

# --- Fonction utilitaire pour créer les Dataloaders ---
def get_oxford_loaders(root_dir, task='contours', batch_size=32, dataset_variant='custom',
                       aug_percent=0.0, aug_rotation_limit=0, blur_radius=0.0, noise_level=0.0,
                       sensitivity_type=None, train_test_split_ratio=0.8, num_workers=2):
    """
    Crée les 3 dataloaders (Train, Val, Test) d'un coup avec options d'augmentation.
    
    Args:
        root_dir (str): Chemin vers le dossier 'data/oxford-iiit-pet'
        task (str): 'contours' ou 'boxes'
        batch_size (int): Taille des batchs
        dataset_variant (str): 'original' ou 'custom'
        aug_percent (float): Pourcentage d'augmentation additive (0.0 à 1.0+)
        aug_rotation_limit (int): Angle max de rotation pour augmentation
        blur_radius (float): Rayon max de flou gaussien
        noise_level (float): Niveau max de bruit gaussien
        sensitivity_type (str): Type de transformation pour test de sensibilité (None, 'flip', 'rotation', 'blur', 'noise')
        train_test_split_ratio (float): Ratio de split train/test (défaut: 0.8 = 80% train, 20% val+test)
    
    Returns:
        dict: Dictionnaire avec les clés 'train', 'val', 'test'
    """
    
    train_ds = OxfordPetDataset(
        root_dir, split='train', task=task, dataset_variant=dataset_variant,
        aug_percent=aug_percent, aug_rotation_limit=aug_rotation_limit,
        blur_radius=blur_radius, noise_level=noise_level,
        train_test_split_ratio=train_test_split_ratio
    )
    
    val_ds = OxfordPetDataset(
        root_dir, split='val', task=task, dataset_variant=dataset_variant,
        train_test_split_ratio=train_test_split_ratio
    )
    
    test_ds = OxfordPetDataset(
        root_dir, split='test', task=task, dataset_variant=dataset_variant,
        blur_radius=blur_radius, noise_level=noise_level, 
        sensitivity_type=sensitivity_type,
        train_test_split_ratio=train_test_split_ratio
    )
    
    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    return loaders