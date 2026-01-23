#!/bin/bash
# Script to train and compare all classification models for bounding box regression

echo "============================================"
echo "  Classification - Model Comparison"
echo "============================================"
echo ""

# Array of config files
configs=(
    "models/config_cnn.yaml"
    "models/config_vgg_finetune.yaml"
    "models/config_vgg_transferlearning.yaml"
)

# Loop through each config and train the model
for config in "${configs[@]}"; do
    echo "--------------------------------------------"
    echo "Training model with config: $config"
    echo "--------------------------------------------"
    python training_classification.py --config "$config"
    echo ""
done