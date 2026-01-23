#!/bin/bash
# Script to train and compare all classification models for bounding box regression

echo "============================================"
echo "  Bounding Box Regression - Model Comparison"
echo "============================================"
echo ""

# Array of config files
configs=(
    "models/config_baseline_cnn_augmented.yaml"
)

# Loop through each config and train the model
for config in "${configs[@]}"; do
    echo "--------------------------------------------"
    echo "Training model with config: $config"
    echo "--------------------------------------------"
    python training_classification.py --config "$config"
    echo ""
done