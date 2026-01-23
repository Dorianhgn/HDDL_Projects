#!/bin/bash
# Script d'entraînement ResNet50 en 2 phases (Head Only → Full Fine-Tuning)

echo "============================================"
echo "  Entraînement ResNet50 en 2 Phases"
echo "  Phase 1 : Head Only (backbone gelé)"
echo "  Phase 2 : Full Fine-Tuning (tout dégelé)"
echo "============================================"
echo ""

# Phase 1 : Head Only (Tête uniquement)
echo "PHASE 1 : Entraînement de la tête uniquement"
echo "   - Backbone : GELÉ (freeze_backbone=True)"
echo "   - LR : 0.001 (élevé)"
echo "   - Epochs : 25"
echo "--------------------------------------------"
python training_classification.py --config models/config_phase1_head_only.yaml

# Vérifier que Phase 1 a bien produit un checkpoint
if [ ! -f "models/temp_results/classification_fine/phase1_head_only/best.pth" ]; then
    echo ""
    echo "ERREUR : Phase 1 n'a pas produit de checkpoint best.pth"
    echo "   Le fichier attendu n'existe pas :"
    echo "   models/temp_results/classification_fine/phase1_head_only/best.pth"
    echo ""
    echo "Abandon de Phase 2."
    exit 1
fi

echo ""
echo "Phase 1 terminée avec succès !"
echo "   Checkpoint sauvegardé : models/temp_results/classification_fine/phase1_head_only/best.pth"
echo ""
echo "============================================"
echo ""

# Phase 2 : Full Fine-Tuning
echo "PHASE 2 : Full Fine-Tuning (tout le réseau)"
echo "   - Backbone : DÉGELÉ (freeze_backbone=False)"
echo "   - LR : 0.00001 (100x plus faible)"
echo "   - Epochs : 40"
echo "   - Chargement des poids de Phase 1"
echo "--------------------------------------------"
python training_classification.py --config models/config_phase2_full_finetuning.yaml

echo ""
echo "============================================"
echo "Entraînement complet terminé !"
echo "============================================"
echo ""
echo "Résultats disponibles dans :"
echo "   - Phase 1 : models/temp_results/classification_fine/phase1_head_only/"
echo "   - Phase 2 : models/temp_results/classification_fine/phase2_full_finetuning/"
echo ""
echo "Modèle final : models/temp_results/classification_fine/phase2_full_finetuning/best.pth"
echo ""
