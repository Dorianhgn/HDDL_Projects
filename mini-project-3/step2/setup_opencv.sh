#!/bin/bash

# Arrêter le script immédiatement en cas d'erreur
set -e

echo "=== 1. Installation des dépendances système ==="
# Nécessaire pour les opérations bas niveau d'OpenCV
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0 libgomp1

echo "=== 2. Nettoyage préventif du container ==="
# On s'assure qu'aucune version pré-installée ne viendra perturber l'installation
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python grad-cam captum
rm -rf /usr/local/lib/python3.10/dist-packages/cv2
rm -rf /usr/local/lib/python3.10/site-packages/cv2

echo "=== 3. Installation de NumPy (La Fondation) ==="
# On force cette version précise avant tout le reste
pip install --force-reinstall numpy==1.24.4

echo "=== 4. Installation des outils (Grad-CAM & Captum) ==="
# ATTENTION : Grad-CAM va installer sa propre version d'opencv-python ici
pip install grad-cam captum

echo "=== 5. Correction du conflit OpenCV ==="
# On supprime la version standard installée par Grad-CAM...
pip uninstall -y opencv-python
# ...et on installe NOTRE version Headless, sans utiliser le cache
pip install --no-cache-dir opencv-python-headless==4.8.0.74

echo "=== 6. Vérification Finale ==="
# On vérifie que tout se charge bien ensemble sans erreur de segmentation ou d'import
python3 -c "import numpy as np; import cv2; import pytorch_grad_cam; import captum; print('✅ Succès! NumPy:', np.__version__, '| OpenCV:', cv2.__version__)"

echo "--- INSTALLATION TERMINÉE AVEC SUCCÈS ---"