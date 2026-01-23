# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3.12 (torch)
#     language: python
#     name: hddltorch
# ---

# %%
# from torchvision.datasets import OxfordIIITPet
# import matplotlib.pyplot as plt

# # Cela télécharge les images (environ 800 Mo) automatiquement dans un dossier 'data'
# # target_types='segmentation' télécharge aussi les masques pour la segmentation
# dataset = OxfordIIITPet(root='./data', split='trainval', target_types='segmentation', download=True)

# print("Téléchargement terminé !")

# # Vérification : Afficher un masque pour voir si ça marche
# mask = dataset[0][1] # Récupère le premier masque
# plt.imshow(mask) 
# plt.show()

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


# %% [markdown]
# # Analyse Exploratoire

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

# %% [markdown]
# ## Répartition sur tout le dataset
#
# Tout le dataset = `list.txt`

# %%
import re

def parse_list_file(file_path):
    """
    Lit le fichier list.txt du dataset Oxford-IIIT Pet
    Format des lignes : Image_Name CLASS-ID SPECIES BREED-ID
    
    Returns:
        pd.DataFrame avec colonnes: filename, breed, class_id, species, species_name, breed_id
    """
    data = []
    
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
            # On utilise une regex pour trouver le pattern: texte_chiffre(s) à la fin
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

# Charger le dataset complet
full_dataset_df = parse_list_file(annotations_folder + 'list.txt')

print(f"Dataset complet: {len(full_dataset_df)} images")
print(f"\nNombre de races: {full_dataset_df['breed'].nunique()}")
print(f"Nombre de classes: {full_dataset_df['class_id'].nunique()}")
print(f"\nPremières lignes:")
print(full_dataset_df.head(10))

# %%
import matplotlib.pyplot as plt

# Répartition par espèce
print("=" * 50)
print("RÉPARTITION PAR ESPÈCE")
print("=" * 50)
species_counts = full_dataset_df['species_name'].value_counts()
print(species_counts)
print(f"\nProportion:")
print(species_counts / len(full_dataset_df) * 100)

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Graphique 1: Répartition par espèce
species_counts.plot(kind='bar', ax=axes[0], color=['#FF6B6B', '#4ECDC4'])
axes[0].set_title('Répartition par Espèce', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Espèce')
axes[0].set_ylabel('Nombre d\'images')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

# Graphique 2: Répartition par race (top 20)
breed_counts = full_dataset_df['breed'].value_counts().head(20)
breed_counts.plot(kind='barh', ax=axes[1], color='#95E1D3')
axes[1].set_title('Top 20 Races (nombre d\'images)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Nombre d\'images')
axes[1].set_ylabel('Race')

plt.tight_layout()
plt.show()

# %%
# Statistiques détaillées par race
print("=" * 50)
print("RÉPARTITION PAR RACE")
print("=" * 50)

breed_stats = full_dataset_df.groupby(['breed', 'species_name']).size().reset_index(name='count')
breed_stats = breed_stats.sort_values('count', ascending=False)

print(f"\nNombre total de races: {len(breed_stats)}")
print(f"  - Races de chats: {len(breed_stats[breed_stats['species_name'] == 'Cat'])}")
print(f"  - Races de chiens: {len(breed_stats[breed_stats['species_name'] == 'Dog'])}")

print(f"\nStatistiques du nombre d'images par race:")
print(f"  - Minimum: {breed_stats['count'].min()}")
print(f"  - Maximum: {breed_stats['count'].max()}")
print(f"  - Moyenne: {breed_stats['count'].mean():.2f}")
print(f"  - Médiane: {breed_stats['count'].median():.2f}")

print(f"\nToutes les races avec leur nombre d'images:")
print(breed_stats.to_string(index=False))

# %%
# Visualisation comparative: Chats vs Chiens par race
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Races de chats
cat_breeds = breed_stats[breed_stats['species_name'] == 'Cat'].sort_values('count', ascending=True)
cat_breeds.plot(x='breed', y='count', kind='barh', ax=axes[0], legend=False, color='#FF6B6B')
axes[0].set_title('Races de Chats (nombre d\'images)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Nombre d\'images')
axes[0].set_ylabel('Race')

# Races de chiens
dog_breeds = breed_stats[breed_stats['species_name'] == 'Dog'].sort_values('count', ascending=True)
dog_breeds.plot(x='breed', y='count', kind='barh', ax=axes[1], legend=False, color='#4ECDC4')
axes[1].set_title('Races de Chiens (nombre d\'images)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Nombre d\'images')
axes[1].set_ylabel('Race')

plt.tight_layout()
plt.show()

# %% [markdown]
#

# %% [markdown]
# ## Vérifier la cohérence et la qualité des masques de segmentation

# %%
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def parse_xml_annotation(xml_path):
    """
    Parse un fichier XML pour extraire les informations de bounding box
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Extraire les informations de l'image
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # Extraire la bounding box
    obj = root.find('object')
    bbox = obj.find('bndbox')
    
    xmin = int(bbox.find('xmin').text)
    ymin = int(bbox.find('ymin').text)
    xmax = int(bbox.find('xmax').text)
    ymax = int(bbox.find('ymax').text)
    
    filename = root.find('filename').text
    animal_name = obj.find('name').text
    
    return {
        'filename': filename,
        'animal': animal_name,
        'width': width,
        'height': height,
        'bbox': (xmin, ymin, xmax, ymax)
    }

def load_trimap(trimap_path):
    """
    Charge un masque trimap
    Valeurs: 1=Foreground (animal), 2=Background, 3=Not classified (contour)
    """
    return Image.open(trimap_path)

# Vérification automatique: Comparer les dimensions et vérifier l'existence des fichiers
print("=" * 60)
print("VÉRIFICATION AUTOMATIQUE DE LA COHÉRENCE")
print("=" * 60)

xmls_folder = annotations_folder + 'xmls/'
trimaps_folder = annotations_folder + 'trimaps/'

# Prendre un échantillon d'images pour vérifier
sample_images = full_dataset_df.sample(n=100, random_state=42)

issues = []
for idx, row in sample_images.iterrows():
    image_name = row['filename'].replace('.jpg', '')
    
    xml_path = xmls_folder + image_name + '.xml'
    trimap_path = trimaps_folder + image_name + '.png'
    image_path = images_folder + row['filename']
    
    # Vérifier l'existence des fichiers
    xml_exists = os.path.exists(xml_path)
    trimap_exists = os.path.exists(trimap_path)
    image_exists = os.path.exists(image_path)
    
    if not xml_exists:
        issues.append(f"XML manquant: {image_name}")
    if not trimap_exists:
        issues.append(f"Trimap manquant: {image_name}")
    if not image_exists:
        issues.append(f"Image manquante: {image_name}")
    
    # Si tous les fichiers existent, vérifier la cohérence des dimensions
    if xml_exists and trimap_exists and image_exists:
        try:
            xml_info = parse_xml_annotation(xml_path)
            trimap = load_trimap(trimap_path)
            image = Image.open(image_path)
            
            # Vérifier que les dimensions correspondent
            if xml_info['width'] != image.width or xml_info['height'] != image.height:
                issues.append(f"Dimensions XML ≠ Image pour {image_name}")
            
            if trimap.size != image.size:
                issues.append(f"Dimensions Trimap ≠ Image pour {image_name}")
                
        except Exception as e:
            issues.append(f"Erreur lors du traitement de {image_name}: {str(e)}")

print(f"\nÉchantillon vérifié: {len(sample_images)} images")
print(f"Nombre de problèmes détectés: {len(issues)}")

if issues:
    print("\nProblèmes détectés:")
    for issue in issues[:10]:  # Afficher les 10 premiers
        print(f"  - {issue}")
    if len(issues) > 10:
        print(f"  ... et {len(issues) - 10} autres problèmes")
else:
    print("\n✓ Aucun problème détecté! Tous les fichiers sont cohérents.")

# Sauvegarder les issues dans un fichier texte
with open('data/coherence_issues.txt', 'w') as f:
    for issue in issues:
        f.write(issue + '\n')


# %%
# Analyse des valeurs dans les trimaps
print("\n" + "=" * 60)
print("ANALYSE DES MASQUES TRIMAP")
print("=" * 60)

# Charger quelques trimaps pour analyser la distribution des valeurs
sample_trimaps = []
for i in range(10):
    row = full_dataset_df.iloc[i]
    image_name = row['filename'].replace('.jpg', '')
    trimap_path = trimaps_folder + image_name + '.png'
    
    if os.path.exists(trimap_path):
        trimap = np.array(load_trimap(trimap_path))
        sample_trimaps.append(trimap)

if sample_trimaps:
    # Analyser les valeurs uniques
    all_values = set()
    for trimap in sample_trimaps:
        all_values.update(np.unique(trimap))
    
    print(f"\nValeurs trouvées dans les trimaps: {sorted(all_values)}")
    print(f"  1 = Foreground (Animal)")
    print(f"  2 = Background (Fond)")
    print(f"  3 = Not classified (Zone d'incertitude entre les deux, i.e zone de contour)")
    
    # Statistiques moyennes
    fg_percentages = []
    bg_percentages = []
    nc_percentages = []
    
    for trimap in sample_trimaps:
        total_pixels = trimap.size
        fg_percentages.append(np.sum(trimap == 1) / total_pixels * 100)
        bg_percentages.append(np.sum(trimap == 2) / total_pixels * 100)
        nc_percentages.append(np.sum(trimap == 3) / total_pixels * 100)
    
    print(f"\nStatistiques moyennes sur {len(sample_trimaps)} masques:")
    print(f"  Foreground (Animal): {np.mean(fg_percentages):.2f}% ± {np.std(fg_percentages):.2f}% de l'image")
    print(f"  Background: {np.mean(bg_percentages):.2f}% ± {np.std(bg_percentages):.2f}% de l'image")
    print(f"  Not classified: {np.mean(nc_percentages):.2f}% ± {np.std(nc_percentages):.2f}% de l'image")


# %% [markdown]
# ### Vérification Manuelle Visuelle
#
# Affichage d'exemples avec:
# 1. Image originale + Bounding Box (XML)
# 2. Image originale + Masque Trimap superposé (transparent)

# %%
def visualize_annotation(image_name, show_breed=True):
    """
    Affiche une image avec sa bounding box et son masque trimap
    """
    # Chemins des fichiers
    xml_path = xmls_folder + image_name + '.xml'
    trimap_path = trimaps_folder + image_name + '.png'
    image_path = images_folder + image_name + '.jpg'
    
    # Vérifier l'existence
    if not all([os.path.exists(p) for p in [xml_path, trimap_path, image_path]]):
        print(f"Fichiers manquants pour {image_name}")
        return
    
    # Charger les données
    xml_info = parse_xml_annotation(xml_path)
    image = Image.open(image_path)
    trimap = np.array(load_trimap(trimap_path))
    
    # Créer la figure avec 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Image originale avec Bounding Box
    axes[0].imshow(image)
    xmin, ymin, xmax, ymax = xml_info['bbox']
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             linewidth=3, edgecolor='red', facecolor='none')
    axes[0].add_patch(rect)
    axes[0].set_title(f'Image + Bounding Box\n{xml_info["animal"].capitalize()}', 
                     fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Image originale avec masque trimap superposé (transparent)
    axes[1].imshow(image)
    
    # Créer un masque coloré
    # 1 = Foreground (vert), 2 = Background (rouge), 3 = Not classified (bleu)
    mask_colored = np.zeros((*trimap.shape, 4))
    mask_colored[trimap == 1] = [0, 1, 0, 0.4]  # Vert transparent pour l'animal
    mask_colored[trimap == 2] = [1, 0, 0, 0.2]  # Rouge transparent pour le fond
    mask_colored[trimap == 3] = [0, 0, 1, 0.5]  # Bleu transparent pour incertain
    
    axes[1].imshow(mask_colored)
    axes[1].set_title('Image + Masque Trimap Superposé\n(Vert=Animal, Rouge=Fond, Bleu=Contour)', 
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Masque seul
    axes[2].imshow(trimap, cmap='viridis')
    axes[2].set_title('Masque Trimap Seul\n(1=Animal, 2=Fond, 3=Contour)', 
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Titre général
    if show_breed:
        breed = image_name.rsplit('_', 1)[0]
        fig.suptitle(f'Race: {breed.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.show()
    
    # Afficher les stats
    total_pixels = trimap.size
    print(f"\nStatistiques pour {image_name}:")
    print(f"  Dimensions: {image.width}x{image.height}")
    print(f"  Bounding Box: ({xmin}, {ymin}) -> ({xmax}, {ymax})")
    print(f"  Pixels Foreground (Animal): {np.sum(trimap == 1)} ({np.sum(trimap == 1)/total_pixels*100:.2f}%)")
    print(f"  Pixels Background: {np.sum(trimap == 2)} ({np.sum(trimap == 2)/total_pixels*100:.2f}%)")
    print(f"  Pixels Not Classified: {np.sum(trimap == 3)} ({np.sum(trimap == 3)/total_pixels*100:.2f}%)")
    print("-" * 60)

# Exemples de visualisation
# Choisir quelques images intéressantes (chat et chien)
examples = [
    'Abyssinian_1',      # Chat
    'american_bulldog_172',  # Chien
]


for example in examples:
    visualize_annotation(example)

# %%
# Visualiser des exemples aléatoires pour une vérification plus complète
print("\n" + "=" * 60)
print("EXEMPLES ALÉATOIRES SUPPLÉMENTAIRES")
print("=" * 60)

# Prendre 3 chats et 3 chiens au hasard
random_cats = full_dataset_df[full_dataset_df['species_name'] == 'Cat'].sample(n=3, random_state=42)
random_dogs = full_dataset_df[full_dataset_df['species_name'] == 'Dog'].sample(n=3, random_state=42)

print("\n🐱 EXEMPLES DE CHATS:")
for _, row in random_cats.iterrows():
    image_name = row['filename'].replace('.jpg', '')
    visualize_annotation(image_name)

print("\n🐶 EXEMPLES DE CHIENS:")
for _, row in random_dogs.iterrows():
    image_name = row['filename'].replace('.jpg', '')
    visualize_annotation(image_name)

# %% [markdown]
# ### Notes (Dorian)
#
# * Voir ce qu'on fait avec les fichiers xml manquants. Perso je vois le truc comme ça :
#   * Dataset global pour classification binaire, classification fine (reconnaitre les 57 races), segmentation par masques (car pas de masques manquants je crois)
#   * Dataset selectif (on retire ceux sans xml) pour faire de la segmentation des bounding boxes.
# * Les images font pas toutes la même taille (500x333, 500x375, 375x500). Donc faut voir comment on donne ça à notre réseau :
#   * Perso, je vois de faire des images genre 500x500 et compléter avec le pixel le plus proche comme en TP.
#   * Ou alors des images 500x375 (si c'est le max genre) et rotater les images qui sont en 375x500. 
# * Pour la prédiction, pas oublier de faire de l'inpainting et tout.

# %%
from dataloader_segmentation import get_oxford_loaders


loaders = get_oxford_loaders(root_dir='data/oxford-iiit-pet/', task='classification', batch_size=16, dataset_variant='custom')

train_df = loaders['train'].dataset.df
train_df

val_df = loaders['val'].dataset.df
val_df.head()

test_df = loaders['test'].dataset.df
test_df.head()

# %%
# Comparaison: Pour chaque race, voir combien d'images dans chaque split
breed_comparison = pd.DataFrame({
    'Total': full_dataset_df['breed'].value_counts(),
    'Train': train_df['breed'].value_counts(),
    'Val': val_df['breed'].value_counts(),
    'Test': test_df['breed'].value_counts()
}).fillna(0).astype(int)

breed_comparison['Train_%'] = (breed_comparison['Train'] / breed_comparison['Total'] * 100).round(1)
breed_comparison['Val_%'] = (breed_comparison['Val'] / breed_comparison['Total'] * 100).round(1)
breed_comparison['Test_%'] = (breed_comparison['Test'] / breed_comparison['Total'] * 100).round(1)

breed_comparison = breed_comparison.sort_values('Total', ascending=False)

print("="*80)
print("RÉPARTITION DÉTAILLÉE PAR RACE (nombre et pourcentage)")
print("="*80)
print(breed_comparison.to_string())

# Statistiques sur les proportions
print("\n" + "="*80)
print("STATISTIQUES DES PROPORTIONS")
print("="*80)
print(f"Proportion moyenne Train: {breed_comparison['Train_%'].mean():.1f}% ± {breed_comparison['Train_%'].std():.1f}%")
print(f"Proportion moyenne Val: {breed_comparison['Val_%'].mean():.1f}% ± {breed_comparison['Val_%'].std():.1f}%")
print(f"Proportion moyenne Test: {breed_comparison['Test_%'].mean():.1f}% ± {breed_comparison['Test_%'].std():.1f}%")

# %%
# Visualisation détaillée: nombre d'images par race dans chaque split
fig, axes = plt.subplots(3, 1, figsize=(16, 15))

for idx, (name, df) in enumerate([('Train', train_df), ('Val', val_df), ('Test', test_df)]):
    breed_counts = df['breed'].value_counts().sort_values(ascending=True)
    breed_counts.plot(kind='barh', ax=axes[idx], color='#95E1D3')
    axes[idx].set_title(f'{name} - Répartition par Race ({len(breed_counts)} races)', 
                       fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('Nombre d\'images')
    axes[idx].set_ylabel('Race')

plt.tight_layout()
plt.show()

# %%
# Répartition des races dans chaque split
print("="*60)
print("RÉPARTITION PAR RACE")
print("="*60)

for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    breed_counts = df['breed'].value_counts()
    print(f"\n{name}:")
    print(f"  Nombre de races différentes: {df['breed'].nunique()}")
    print(f"  Total images: {len(df)}")
    print(f"  Images par race (min/max/moyenne): {breed_counts.min()}/{breed_counts.max()}/{breed_counts.mean():.2f}")

# Vérifier la présence de toutes les races dans chaque split
all_breeds = set(full_dataset_df['breed'].unique())
train_breeds = set(train_df['breed'].unique())
val_breeds = set(val_df['breed'].unique())
test_breeds = set(test_df['breed'].unique())

print("\n" + "="*60)
print("COUVERTURE DES RACES")
print("="*60)
print(f"Total races dans le dataset: {len(all_breeds)}")
print(f"Races dans Train: {len(train_breeds)} ({len(train_breeds)/len(all_breeds)*100:.1f}%)")
print(f"Races dans Val: {len(val_breeds)} ({len(val_breeds)/len(all_breeds)*100:.1f}%)")
print(f"Races dans Test: {len(test_breeds)} ({len(test_breeds)/len(all_breeds)*100:.1f}%)")

# Races manquantes dans certains splits
print(f"\nRaces absentes du Train: {all_breeds - train_breeds if all_breeds - train_breeds else 'Aucune'}")
print(f"Races absentes du Val: {all_breeds - val_breeds if all_breeds - val_breeds else 'Aucune'}")
print(f"Races absentes du Test: {all_breeds - test_breeds if all_breeds - test_breeds else 'Aucune'}")

# %%
# Répartition des espèces dans chaque split
print("="*60)
print("RÉPARTITION PAR ESPÈCE")
print("="*60)

for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    print(f"\n{name}:")
    species_counts = df['species_name'].value_counts()
    print(species_counts)
    print(f"Proportion:")
    print((species_counts / len(df) * 100).round(2))

# Visualisation
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, df) in enumerate([('Train', train_df), ('Val', val_df), ('Test', test_df)]):
    species_counts = df['species_name'].value_counts()
    species_counts.plot(kind='bar', ax=axes[idx], color=['#FF6B6B', '#4ECDC4'])
    axes[idx].set_title(f'{name} - Répartition par Espèce', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Espèce')
    axes[idx].set_ylabel('Nombre d\'images')
    axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.show()

# %%
from dataloader_segmentation import OxfordPetDataset

# Charger les 3 datasets avec l'option custom (par défaut)
root_dir = 'data/oxford-iiit-pet/'

train_dataset = OxfordPetDataset(root_dir, split='train', task='contours', dataset_variant='custom')
val_dataset = OxfordPetDataset(root_dir, split='val', task='contours', dataset_variant='custom')
test_dataset = OxfordPetDataset(root_dir, split='test', task='contours', dataset_variant='custom')

print("Tailles des datasets:")
print(f"Train: {len(train_dataset)}")
print(f"Val: {len(val_dataset)}")
print(f"Test: {len(test_dataset)}")
print(f"Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")

# Accéder aux DataFrames pour analyser
train_df = train_dataset.df
val_df = val_dataset.df
test_df = test_dataset.df

print("\n" + "="*60)
print("Aperçu des données Train:")
print(train_df.head())

# %% [markdown]
# # Test du nouveau DataLoader avec l'option custom
#
# Vérification de la répartition des races dans les différents splits (train/val/test)

# %% [markdown]
# # Test de la Data Augmentation
#
# Comparaison des tailles du dataset train avec et sans augmentation

# %%
from dataloader_segmentation import get_oxford_loaders

print("="*70)
print("TEST DE LA DATA AUGMENTATION")
print("="*70)

# 1. Dataset SANS augmentation (aug_percent=0.0)
print("\n1️⃣  Dataset SANS augmentation (aug_percent=0.0):")
print("-"*70)
loaders_no_aug = get_oxford_loaders(
    root_dir='data/oxford-iiit-pet/', 
    task='contours', 
    batch_size=16, 
    dataset_variant='custom',
    aug_percent=0.0  # Pas d'augmentation
)

train_size_no_aug = len(loaders_no_aug['train'].dataset)
val_size_no_aug = len(loaders_no_aug['val'].dataset)
test_size_no_aug = len(loaders_no_aug['test'].dataset)

print(f"   Train size: {train_size_no_aug}")
print(f"   Val size: {val_size_no_aug}")
print(f"   Test size: {test_size_no_aug}")

# 2. Dataset AVEC augmentation 50% (aug_percent=0.5)
print("\n2️⃣  Dataset AVEC augmentation 50% (aug_percent=0.5):")
print("-"*70)
loaders_aug_50 = get_oxford_loaders(
    root_dir='data/oxford-iiit-pet/', 
    task='contours', 
    batch_size=16, 
    dataset_variant='custom',
    aug_percent=0.5,  # Ajoute 50% d'images augmentées
    aug_rotation_limit=30,
    blur_radius=2.0,
    noise_level=0.1
)

train_size_aug_50 = len(loaders_aug_50['train'].dataset)
val_size_aug_50 = len(loaders_aug_50['val'].dataset)
test_size_aug_50 = len(loaders_aug_50['test'].dataset)

print(f"   Train size: {train_size_aug_50}")
print(f"   Val size: {val_size_aug_50}")
print(f"   Test size: {test_size_aug_50}")

# 3. Dataset AVEC augmentation 100% (aug_percent=1.0)
print("\n3️⃣  Dataset AVEC augmentation 100% (aug_percent=1.0):")
print("-"*70)
loaders_aug_100 = get_oxford_loaders(
    root_dir='data/oxford-iiit-pet/', 
    task='contours', 
    batch_size=16, 
    dataset_variant='custom',
    aug_percent=1.0,  # Double le dataset
    aug_rotation_limit=30,
    blur_radius=2.0,
    noise_level=0.1
)

train_size_aug_100 = len(loaders_aug_100['train'].dataset)
val_size_aug_100 = len(loaders_aug_100['val'].dataset)
test_size_aug_100 = len(loaders_aug_100['test'].dataset)

print(f"   Train size: {train_size_aug_100}")
print(f"   Val size: {val_size_aug_100}")
print(f"   Test size: {test_size_aug_100}")

# Résumé comparatif
print("\n" + "="*70)
print("RÉSUMÉ COMPARATIF")
print("="*70)

comparison_df = pd.DataFrame({
    'Split': ['Train', 'Val', 'Test'],
    'Sans Aug': [train_size_no_aug, val_size_no_aug, test_size_no_aug],
    'Aug 50%': [train_size_aug_50, val_size_aug_50, test_size_aug_50],
    'Aug 100%': [train_size_aug_100, val_size_aug_100, test_size_aug_100]
})

comparison_df['Gain 50%'] = comparison_df['Aug 50%'] - comparison_df['Sans Aug']
comparison_df['Gain 100%'] = comparison_df['Aug 100%'] - comparison_df['Sans Aug']

print(comparison_df.to_string(index=False))

print("\n✅ Constat:")
print(f"   - Val et Test ne sont JAMAIS augmentés (aug_percent s'applique uniquement au train)")
print(f"   - Train avec aug_percent=0.5 : +{train_size_aug_50 - train_size_no_aug} images ({(train_size_aug_50/train_size_no_aug - 1)*100:.1f}% d'augmentation)")
print(f"   - Train avec aug_percent=1.0 : +{train_size_aug_100 - train_size_no_aug} images ({(train_size_aug_100/train_size_no_aug - 1)*100:.1f}% d'augmentation)")

# %%
# Visualisation comparative
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(comparison_df['Split']))
width = 0.25

bars1 = ax.bar(x - width, comparison_df['Sans Aug'], width, label='Sans Aug', color='#FF6B6B')
bars2 = ax.bar(x, comparison_df['Aug 50%'], width, label='Aug 50%', color='#4ECDC4')
bars3 = ax.bar(x + width, comparison_df['Aug 100%'], width, label='Aug 100%', color='#95E1D3')

ax.set_xlabel('Split', fontweight='bold')
ax.set_ylabel('Nombre d\'images', fontweight='bold')
ax.set_title('Comparaison des tailles de dataset avec/sans Data Augmentation', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Split'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Ajouter les valeurs sur les barres
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualisation d'exemples d'images augmentées
#
# Comparaison image originale vs images avec différentes transformations

# %%
import torch
import matplotlib.pyplot as plt

# Prendre un échantillon du dataset avec augmentation
train_dataset_aug = loaders_aug_50['train'].dataset

# Indice d'une image originale
original_idx = 0

# Indices d'images augmentées (même image mais transformée)
# Les images augmentées commencent à l'index original_len
augmented_idx = train_dataset_aug.original_len + original_idx

print(f"Dataset info:")
print(f"  Nombre d'images originales: {train_dataset_aug.original_len}")
print(f"  Nombre d'images augmentées: {train_dataset_aug.augmented_len}")
print(f"  Total: {train_dataset_aug.total_len}")
print(f"\nAffichage de l'image #{original_idx}:")

# Récupérer l'image originale et une version augmentée
img_orig, mask_orig = train_dataset_aug[original_idx]
img_aug, mask_aug = train_dataset_aug[augmented_idx]

# Fonction helper pour convertir tensor en image affichable
def tensor_to_img(tensor):
    return tensor.permute(1, 2, 0).cpu().numpy()

# Affichage
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Image originale
axes[0, 0].imshow(tensor_to_img(img_orig))
axes[0, 0].set_title('Image Originale', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Masque original
axes[0, 1].imshow(mask_orig.cpu().numpy(), cmap='viridis')
axes[0, 1].set_title('Masque Original', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

# Image augmentée
axes[1, 0].imshow(tensor_to_img(img_aug))
axes[1, 0].set_title('Image Augmentée (transformation aléatoire)', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

# Masque augmenté
axes[1, 1].imshow(mask_aug.cpu().numpy(), cmap='viridis')
axes[1, 1].set_title('Masque Augmenté', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

plt.suptitle(f'Comparaison Original vs Augmenté (même image source)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n💡 Note: Les transformations sont appliquées aléatoirement parmi:")
print("   - Flip horizontal")
print("   - Rotation (±30°)")
print("   - Flou gaussien (image seulement)")
print("   - Bruit gaussien (image seulement)")

# %% [markdown]
#

# %% [markdown]
# ## Test du mapping breed_id -> label pour classification fine-grained
#
# Vérification que les labels sont bien de 0 à 36 (25 chats + 12 chiens)

# %%
from dataloader_segmentation import get_oxford_loaders

# Charger le dataloader pour classification fine-grained
loaders_classification_fine = get_oxford_loaders(
    root_dir='data/oxford-iiit-pet/', 
    task='classification_fine',
    batch_size=32, 
    dataset_variant='custom'
)

# Récupérer le dataset depuis le dataloader
train_dataset = loaders_classification_fine['train'].dataset
test_dataset = loaders_classification_fine['test'].dataset

# Extraire tous les labels du dataset d'entrainement
all_labels = []
for i in range(len(test_dataset)):
    _, label = test_dataset[i]
    all_labels.append(label.item())


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Convertir en numpy array pour les statistiques
all_labels = np.array(all_labels)

print("=" * 60)
print("STATISTIQUES DES LABELS - CLASSIFICATION FINE-GRAINED")
print("=" * 60)

print(f"Nombre total d'échantillons: {len(all_labels)}")
print(f"Labels uniques: {len(np.unique(all_labels))}")
print(f"Plage de labels: {all_labels.min()} à {all_labels.max()}")

# Distribution des labels
label_counts = pd.Series(all_labels).value_counts().sort_index()
print(f"\nDistribution des labels:")
print(label_counts)

# Statistiques
print(f"\nStatistiques:")
print(f"  - Moyenne: {all_labels.mean():.2f}")
print(f"  - Médiane: {np.median(all_labels):.2f}")
print(f"  - Écart-type: {all_labels.std():.2f}")
print(f"  - Min/Max occurrences par label: {label_counts.min()}/{label_counts.max()}")
print(f"  - Moyenne d'images par label: {label_counts.mean():.2f}")

# Visualisation
plt.figure(figsize=(15, 6))
label_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution des Labels - Classification Fine-Grained', fontweight='bold', fontsize=14)
plt.xlabel('Label (Breed ID)')
plt.ylabel('Nombre d\'images')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Vérification de la cohérence avec les races
print(f"\n✅ Vérification: Nous avons {len(np.unique(all_labels))} labels uniques")
print(f"   Le dataset Oxford-IIIT Pet contient 37 races au total (25 chats + 12 chiens)")
print(f"   Labels attendus: 0 à 36")

# %%
