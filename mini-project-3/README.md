# ViT vs CNN

Dans ce projet, on va tester la performance de deux architectures de réseaux de neurones populaires pour la classification d'images : les Vision Transformers (ViT) et les Convolutional Neural Networks (CNN). Nous allons utiliser le dataset CIFAR-10 pour entraîner et évaluer les modèles.

## Étapes du projet

Chaque personne va developper une étape du projet. Je pense que faire **des branches GIT** et genre un sous dossier est idéal (`step1/`, `step2`, etc.)

Un repo du genre (avec tous les notebooks personnels sous mini_project_3/ directement pour éviter les pb après quand on combine dans `notebook_combined.ipynb`).

```
HDDL_PROJECTS/
├── mini_project_1/
│       └── ...
├── mini_project_2/
│       └── ...
├── mini_project_3/
│   ├── notebook1.ipynb
│   ├── step1/              # FASHION MNIST - MODEL HANDMADE
│   │   ├── ViT.py                  # ViT by hand (Lise)
│   │   ├── CNN.py                  # CNN by hand (with Conv2d...)
│   │   ├── dataloader.py           # Train, val, test dataloaders
│   │   ├── train.py                # Training (save models_weights, losses...)
│   │   ├── test.py
│   │   ├── data/                   # MNIST data
│   │   └── ...
│   ├── notebook2.ipynb
│   ├── step2/              # VIT >> CNN LORS DE REASONING TASKS
│   │   ├── models.py               # ViT et ResNet de torch
│   │   ├── train.py                # Training (save models_weights, losses...)
│   │   ├── dataloader.py           # Solution 1 & Solution 2 ci-dessous
│   │   ├── data/                   # CLEVR data
│   │   └── ...
│   ├── notebook3.ipynb
│   ├── step3/              # VIT >> CNN LORS DE DONNÉES TRANSFORMÉES
│   │   ├── models.py               # ResNet & ViT équivalents (petits) pré-entrainé sur ImageNet
│   │   ├── finetune.py             # Fine-Tuning (save models_weights, losses...)
│   │   ├── dataloader.py           # Images permutées (train, val, test)
│   │   ├── data/                   # CIFAR100 data
│   │   └── ...
│   ├── notebook4.ipynb
│   ├── step4/              # SOTA MODELS (ConvNeXt, Swin)
│   │   ├── models.py/              # ConvNeXt, Swin petits préentrainés sur ImageNet
│   │   ├── dataloader.py           # Subset de ImageNet-R
│   │   ├── finetune.py             # Fine-Tuning (save models_weights, losses...)
│   │   ├── data/                   # Une partie de ImageNet-R data (1k ?)
│   │   └── ...
│   └── notebook_final.ipynb        # Notebook combiné
├── README.md
├── requirements.txt
└── .gitignore
``` 

### 1. Fashion MNIST (From Scratch)

- Developper à la main un **petit CNN** (3 couches `Conv2d`) vs un **ViT-Lite** (Patch size 4 ou 7, 4 têtes d'attention) et les entraîner sur **Fashion MNIST**.
- **Ce qu'il faut montrer** : Courbes de Loss/Accuracy. Le ViT va "osciller" et mettre 3x plus de temps à atteindre 80% alors que le CNN y va en ligne droite.
- **Interprétation** : Afficher les **Position Embeddings** du ViT à l'époque 1 (aléatoires) vs l'époque 50 (on commence à voir une structure de grille). Pour le CNN, montre les poids de la `conv1` : ils ressemblent déjà à des détecteurs de bords.

### 2. Raisonnement (CLEVR)

#### **Modèles** : un ViT et un CNN.

#### **Datset CLEVR**

**Solution 1: "Filtrage Conditionnel"**

- On demande **"Quelle est la couleur du CYLINDRE le plus loin ?**

- Ca casse le CNN car il doit donc combiner détection de forme (feature locale) + comparaison de position (globale). S'il se contente de scanner le haut de l'image, il se trompe. Le ViT sera meilleur sur cette tâche.

**Solution 2: "La "Distance Relative" (Mode Hardcore)"**

- Pour torturer le CNN, la tâche est : **"Quelle est la couleur de l'objet le plus proche de la sphère grise ?"**

    - La "sphère grise" (l'ancre) peut être n'importe où : en haut, en bas, à gauche...

    - Le modèle doit d'abord trouver l'ancre, puis scanner les alentours (attention globale) pour trouver le voisin le plus proche.

    - Il n'y a aucun biais de position. "En haut" ne veut rien dire ici.

    - Le ViT excelle ici car son mécanisme d'attention Query−Key est littéralement conçu pour calculer des similarités entre positions.

> Pour créer ce dataset, on trie d'abord les images contenant **une seule** sphère grise (peu importe son matériel), et on récupère les coordonnées les plus proches de cet objet.

### 3. Le Challenge "Puzzle" (Permuted CIFAR/ImageNet)

- **Modèles** : ResNet-18 vs ViT-Tiny (pré-entraînés sur ImageNet).
- **Méthode** : Appliquer la même permutation de patchs (ex: grille 4x4 mélangée) sur tout ton set de test.
- **Fine Tuner** le modèle très légèrement (5 epoch, lr=1e-6)
- **Resultat** : Le ResNet va perdre peut-être 40% d'accuracy, le ViT seulement 5-10% après un très léger fine-tuning des embeddings. C'est ta preuve ultime que le ViT comprend la structure globale ("l'image est un ensemble de patchs") alors que le CNN est esclave de la topologie locale. 
- **Interpretabilité** : 
    - On peut réutiliser (1) pour voir que c'est le PE qui sont appris à nouveau dans le ViT.
    - Pour le CNN, on utilise **Grad_CAM** avant et après la permutation. On verra que le CNN "s'allume" sur des zones totalement aléatoires ou sur les bordures des patchs (le bruit haute fréquence créé par la coupure), sans jamais réussir à isoler l'objet.

### 4. La guerre des SOTA (ConvNeXt vs Swin)

- **Modèles** : ConvNeXt et Swin préentrainées sur ImageNet.
- **Dataset : ImageNet-R (Rendition)** : Le choix du "Shape Biais"
- **Méthode** : fine-tuner le modèles très légèrement (5-10 epoch, lr petit).
- **Resultats attendus** :
    - Les **CNN** (même ConvNeXt) sont très dépendants de la texture. Si la texture "photo" disparaît au profit d'un trait de crayon, le CNN se perd.
    - **Swin** : Gagne souvent ici car les Transformers se basent sur la forme globale (shape bias). Un Swin reconnaît un "éléphant" par sa silhouette et la relation entre ses membres, peu importe si la texture est de la peinture ou du fusain.
    - **Interprétabilité** : C'est flagrant. Sur une version "dessin" d'un objet, le **Grad-CAM** de ConvNeXt sera tout éparpillé (il cherche des textures qu'il ne trouve pas), tandis que l'Attention Map de Swin épousera encore parfaitement la forme de l'objet.
- **Exemple à montrer** :
    - Trouver un objet dans le dataset, genre un **Origami** en **Sclupture de glace** et faire le **Grad-CAM**.

> Si y'a le temps, faire la même sur **ImageNet-A** (bruit, blur, lightning, etc.) et techniquement, le **ConvNeXt** serait meilleur avec un léger finetuning.