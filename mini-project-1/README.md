# Projets Chats ou Chiens ?

## Structure

* On doit créer chacun des fichiers .py pour nos modèles.
* Créer deux fichier unifiés (boucle d'entrainement et boucle de test), un pour la classification, un pour la regression avec pleins d'options (genre nombres de classes à prédire, arguments variables pour les modèles).
* Créer les Dataloader dans un fichier .py
* Run tout dans un notebook personnel (qui ne sera pas commit car tous les notebooks sont dans le `.gitignore`) en y important les modules dans les .py
* Donc besoin uniquement que de deux branches : `dev-classification` & `dev-segmentation`. Pas forcément de conflits car chacun bosse sur ses fichiers .py


## A faire

### 1. Analyse exploratoire

* Comprend l'analyse explo basique

* Attention, les images n'ont pas le même format ! Faire qqch pour le dataset

### 2. Classification binaire & Classification Fine

*Lise et Matteo*

* La création des Dataset (Tensor) et des DataLoader pour les 2 taches
    * train (shuffle = true)
    * val et test (shuffle = false)
    * batch_size = 64 ou 128
    * num_workers = 4
    * etc.

* Utiliser ce qu'on a fait en TP ImageClassification, ie :
    * basic convolutional NN avec des dense -> conseil : le définir comme dans le tp ae_ssl>ssl.ipynb avec self.encoder et self.classfier pour justement facilement intégrer ce qu'on a fait en TP ssl.
    * transfer learning à partir du VGG genre (fine tuner à plusieurs étapes)
    * Le coder en pytorch (on l'a fait en Tensorflow en TP).
* Utiliser ce qu'on a fait en TP ae_ssl>ssl.ipynb : **Downstream Task 1: Classification on Mnist dataset**
    * Suivre exactement ce qu'on a fait là
    * Tester avec inpainting encoder et mae_encoder
    * Faire en sorte que l'encoder + classifier soit le même que le "basic convolutional NN avec des dense" d'au dessus pour pouvoir comparer l'effet de *inpainting* et du *mae*

**Faire exactement de même pour la classifcation fine (des races)**

* Donc au final, il y aura dans la branche `dev-classification` :
    * 1 fichier d'entrainement / test pour la classification
    * Des fichiers models_model_name.py sous `models/`
        * 1 fichier = 1 modèle + ses variantes (genre le basicConvNN avec dense, le même avec de l'inpaiting, le même avec le mae.)
    * des notebooks a vos noms (qui ne seront pas commit) pour :
        * tester ces fonctions.
        * répondre aux questions du pdf (genre Abyssinian vs Bengal)
    * des fichiers dataloader_base ou dataloader_fine qui génère le dataloader selon si c'est une classification fine ou pas


### 3. Segmentation des animaux

*Dorian & Sara*

**Deux taches : segmentation des boxes, et segmentation des "contours"**

* La création des Dataset (Tensor) et des DataLoader pour les 2 taches
    * train (shuffle = true)
    * val et test (shuffle = false)
    * batch_size = 64 ou 128
    * num_workers = 4
    * etc.

> Attention, pour les box de segmentation, on a plein de valeurs manquantes, créer un nouveau DataLoader en conséquence.

* Créer le U-Net à la main avec les n_s (nombre de "descentes") :
    * Créer une classe BasicConv2d(nn.module) : Conv2d, batch_norm, ReLU ou LeakyRELU
    * Créer une classe DoubleConv2d(nn.module) : qui est 2x celle d'au dessus
    * Créer une classe Encoder(nn.module) avec :
        * dans l'init : self.enc = nn.Sequential(DoubleConv(C_in, C_hid, stride=1), *[DoubleConv(C_id, C_id, stride = 2) for _ in range (n_s - 1)])
        * dans le forward : 
            * le tableau des skips connections initialisé
            * pour chaque layer de self.enc, on calcule x = layer(x) et on skip.append(x)
            * on renvoie le x final et les skips connection
    * Créer une classe Decoder(nn.module) :
        * nn.dec : (n_s - 1) Double_Conv avec stride = 2
        * nn.final_conv : 1 final Double_Conv avec stride = 1
        * nn.readout = nn.Conv2d(C_hid, **C_out**, 1) -> ce qui permet de convertir dans ce qu'on veut
        * forward (hid, skip):
            * boucler sur la taille de dec (ou n_s - 1)
                * on initialise hid
                * on ajoute les skip connctions qui correspond skip[...] dont ... à def
            * après la boucle, on fait passer hid dans final_conv et readout et on retrourne Y

* Le tester pour les 2 types de segmentations.

* Donc au final, il y aura dans la branche `dev-segmentation` :
    * 1 fichier d'entrainement / test pour la segmentation
    * Des fichiers models_model_name.py sous `models/`
        * 1 fichier = 1 modèle + ses variantes (genre le U-Net et les pistes complémentaires)
    * des notebooks a vos noms (qui ne seront pas commit) pour :
        * tester ces fonctions.
        * répondre aux questions du pdf (evaluer la qualité de la segmentation)
    * des fichiers dataloader_box ou dataloader_contours qui génère le dataloader selon si c'est une segmentation des boxes ou des contours.

### 4. Analyse comparative

### 5. Pistes complémentaires.

Tester chacun nos modèles aux transformations (déjà intégré dans l'application au ssl en un sens)

On en reparlera ensemble pour ce qui est augmentation des données.
