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

---

### 🧩 3.2. Avant de Commencer un Travail (Tirez !)

1. **Mettez à jour** votre branche principale et créez (ou passez à) votre branche de travail :

   ```bash
   git checkout main
   git pull
   git checkout -b dev-[votre_nom]/[nom_du_travail]  # Si vous créez une nouvelle branche
   # OU
   git checkout dev-[votre_nom]/[nom_du_travail]     # Si elle existe déjà
   ```

2. **Rendez les scripts exécutables** (⚠️ à faire une seule fois après avoir cloné le dépôt) :

   ```bash
   chmod +x sync_before_push.sh
   chmod +x sync_after_pull.sh
   ```

3. **Synchronisez les notebooks** pour que vos `.ipynb` locaux soient à jour avec les `.py` de la branche tirée :

   ```bash
   ./sync_after_pull.sh
   ```

   > ✅ Ce script parcourt automatiquement tous les mini-projets (`mini-project-1/`, `mini-project-2/`, `mini-project-3/`, etc.)
   > et met à jour les notebooks `.ipynb` à partir des fichiers `.py` versionnés dans le dépôt.
   >
   > ⚠️ Si vous travaillez pour la première fois sur un fichier et qu’une erreur apparaît, il peut être nécessaire de le *"pairer"* une fois :
   >
   > ```bash
   > jupytext --set-formats py:percent,ipynb mini-project-x/nom_du_fichier.ipynb
   > ```

---

### 🧩 3.3. Après avoir Fini un Travail (Poussez !)

1. **Exécutez** votre notebook `.ipynb` pour générer toutes les **sorties** et les **figures** nécessaires au rendu final.

2. **Synchronisez** les fichiers `.py` avec vos notebooks avant de push :

   ```bash
   ./sync_before_push.sh
   ```

   > 🔄 Ce script convertit automatiquement tous les notebooks `.ipynb` des mini-projets en scripts `.py` au format `py:percent`
   > pour éviter les conflits de merge sur les sorties Jupyter.

3. **Ajoutez, *Committez* et *Pushez* :**

   ```bash
   git add .
   git commit -m "feat: [votre_nom] mise à jour du mini-projet"
   git push -u origin dev-[votre_nom]/[nom_du_travail]
   ```

4. **Ouvrez une *Pull Request (PR)*** depuis votre branche vers `main` sur la plateforme (GitHub/GitLab).
   Demandez à un autre membre de l’équipe de **review** votre code avant la fusion.


-----

## 📂 4. Organisation des Mini-Projets

Chaque sous-dossier correspond à un mini-projet. Veillez à y placer tous les scripts et notebooks associés.

  * `mini-project-1/` : Modèle HDDL CFM/Diffusion initial.
  * `mini-project-2/` : (À créer)
  * `mini-project-3/` : (À créer)
