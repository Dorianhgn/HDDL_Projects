# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: HDDLtorch
#     language: python
#     name: python3
# ---

# %%
from torchvision.datasets import OxfordIIITPet
import matplotlib.pyplot as plt

# Cela télécharge les images (environ 800 Mo) automatiquement dans un dossier 'data'
# target_types='segmentation' télécharge aussi les masques pour la segmentation
dataset = OxfordIIITPet(root='./data', split='trainval', target_types='segmentation', download=True)

print("Téléchargement terminé !")

# Vérification : Afficher un masque pour voir si ça marche
mask = dataset[0][1] # Récupère le premier masque
plt.imshow(mask) 
plt.show()

# %%
path = './data'

# %%
#packages utiles
# Utils
import os
import shutil
import time

# Maths - Stats
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import random as rd

# Data visualization
import seaborn as sns


# %%
# divide dataset into train, val, test
from sklearn.model_selection import train_test_split

# Chemins basés sur ta structure de dossiers
base_path = 'data/oxford-iiit-pet/'
images_folder = base_path + 'images/'
annotations_folder = base_path + 'annotations/'

def parse_annotation_file(file_path):
    """
    Lit les fichiers .txt fournis par Oxford-Pet (trainval.txt ou test.txt)
    Format des lignes : Nom_Image ClassID SpeciesID BreedID
    SpeciesID : 1 = Chat, 2 = Chien
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            # Parfois il y a plusieurs espaces, on nettoie
            parts = [p for p in parts if p] 
            
            filename = parts[0] + ".jpg"
            species_id = int(parts[2]) # 1 = Chat, 2 = Chien
            
            # On convertit en 0 (Chat) et 1 (Chien) comme dans ton TP
            # Si species_id est 1 (Chat) -> devient 0
            # Si species_id est 2 (Chien) -> devient 1
            category = 0 if species_id == 1 else 1
            
            data.append({'filename': filename, 'category': str(category)})
            
    return pd.DataFrame(data)

# 1. Charger les données officielles
# trainval.txt contient TOUT l'entrainement (qu'on va diviser en Train et Validation)
trainval_df = parse_annotation_file(annotations_folder + 'trainval.txt')
test_df = parse_annotation_file(annotations_folder + 'test.txt')

# 2. Séparer trainval en Train (80%) et Validation (20%)
# On utilise une fonction scikit-learn pour mélanger et couper proprement
total_train_df, total_validation_df = train_test_split(
    trainval_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=trainval_df['category'] # Important : Garde la même proportion chat/chien
)

# 3. Affichage des résultats (comme dans ton TP)
print("Dimensions des DataFrames :")
print(f"Train: {total_train_df.shape[0]}")
print(f"Validation: {total_validation_df.shape[0]}")
print(f"Test: {test_df.shape[0]}")

print("\nExemple des premières lignes (Train) :")
print(total_train_df.head())

# %%
# is the dataset balanced?
total_train_df['category'].value_counts().plot(kind='bar', title='Répartition des classes dans le dataset d\'entrainement')


# %%
test_df['category'].value_counts().plot(kind='bar', title='Répartition des classes dans le dataset de test')
