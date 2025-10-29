# 📓 README.md : Projet HDDL AOD Eurasia-Sahara

## 🌟 1. Bienvenue \!

Ce dépôt contient les travaux de modélisation *High-Dimensional Deep Learning* (HDDL) pour la prédiction de la profondeur optique d'aérosols (AOD) sur la zone Eurasie-Sahara. Nous sommes l'équipe : **Lise, Matteo, Sara, et Dorian.**

## ⚙️ 2. Configuration Initiale de l'Environnement

Afin de garantir que tout le monde utilise les mêmes dépendances et le même workflow, veuillez suivre ces étapes **une seule fois** :

### 2.1. Clone du Dépôt

Ouvrez votre terminal et clonez le dépôt :

```bash
git clone [URL_DE_VOTRE_DEPOT]
cd [NOM_DU_DEPOT]
```

### 2.2. Activation de l'Environnement Conda

Nous utilisons l'environnement Conda `HDDLtorch` pour tous les travaux :

```bash
conda activate HDDLtorch
```

### 2.3. Installation des Outils de Développement

Installez **Jupytext** pour la gestion des notebooks et d'autres dépendances utiles si elles ne sont pas déjà dans votre environnement :

```bash
pip install jupytext notebook # installe jupytext et s'assure que les outils de notebooks sont là
pip install -r requirements.txt # si vous avez un fichier requirements
```

**🚨 IMPORTANT :** Si vous n'avez pas encore de fichier `requirements.txt`, créez-le immédiatement à la racine pour lister toutes vos dépendances \!

### 2.4. Configuration Git (Ignorer les Notebooks)

Pour un *versioning* propre, nous **ignorons** le fichier `.ipynb` généré. Seul le fichier source `.py` sera versionné.

Assurez-vous que le fichier `.gitignore` à la racine du projet contient (au minimum) la ligne suivante :

```
# Ignorer tous les fichiers Notebooks générés
*.ipynb
```

-----

## 💻 3. Workflow de Développement

Nous travaillons par *feature branch* pour éviter de casser la branche principale (`main`).

### 3.1. Structure des Branches

  * **`main` :** Contient uniquement le code stable et testé. **Interdit de *push* directement sur `main`.**
  * **`dev-[votre_nom]/[nom_du_travail]` :** Votre branche de travail. Par exemple : `dev-dorian/refactor-cfn` ou `dev-lise/preprocessing-aod`.

### 3.2. Avant de Commencer un Travail (Tirez \!)

1.  **Mettez à jour** votre branche principale et créez (ou passez à) votre branche de travail :

    ```bash
    git checkout main
    git pull
    git checkout -b dev-dorian/mini-project-2 # Si vous créez une nouvelle branche
    # OU
    git checkout dev-dorian/mini-project-2 # Si elle existe déjà
    ```

2.  **Synchronisez** le notebook pour vous assurer que les versions locales `.ipynb` sont à jour avec le code `.py` tiré :

    ```bash
    # Exécutez cette commande pour chaque mini-projet que vous modifiez
    jupytext --to ipynb mini-project-1/modele_aod.py
    ```

    > **Note :** Si vous travaillez pour la première fois sur un fichier, il se peut que vous deviez le "paired" une fois : `jupytext --set-formats py:percent,ipynb mini-project-1/modele_aod.ipynb` (puis recommencer l'étape 2.1).

### 3.3. Après avoir Fini un Travail (Poussez \!)

1.  **Exécutez** le code dans votre notebook `.ipynb` pour générer toutes les **sorties** et les **figures** (nécessaires pour le rendu final).

2.  **Synchronisez** le fichier source `.py` avec les commentaires Markdown/sorties que vous avez pu ajouter dans le `.ipynb` (cela garantit que vos interprétations sont dans le fichier versionné) :

    ```bash
    jupytext --to py:percent mini-project-1/modele_aod.ipynb
    ```

3.  **Ajoutez, *Committez* et *Pushez* :**

    ```bash
    git add mini-project-1/modele_aod.py
    git commit -m "feat: [votre_nom] Ajout du modèle d'attention pour l'AOD"
    git push -u origin dev-dorian/mini-project-2
    ```

4.  **Ouverture de la *Pull Request (PR)*** : Rendez-vous sur la plateforme (GitHub/GitLab) et ouvrez une **Pull Request** de votre branche vers `main`. Demandez à un autre membre de l'équipe de **réviser** (review) votre code avant la fusion.

-----

## 📂 4. Organisation des Mini-Projets

Chaque sous-dossier correspond à un mini-projet. Veillez à y placer tous les scripts et notebooks associés.

  * `mini-project-1/` : Modèle HDDL CFM/Diffusion initial.
  * `mini-project-2/` : (À créer)
  * `mini-project-3/` : (À créer)
