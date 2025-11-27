#!/bin/bash
# Script to train and compare all classification models for bounding box regression

echo "============================================"
echo "  Bounding Box Regression - Model Comparison"
echo "============================================"
echo ""

# Array of config files
configs=(
    "models/config_unet_encoder.yaml"
    "models/config_vgg_head_only.yaml"
    "models/config_vgg_head_last_conv.yaml"
    "models/config_vgg_full_finetune.yaml"
    "models/config_resnet18_head_only.yaml"
    "models/config_resnet18_full_finetune.yaml"
    "models/config_resnet50_head_only.yaml"
)

# Loop through each config and train the model
for config in "${configs[@]}"; do
    echo "--------------------------------------------"
    echo "Training model with config: $config"
    echo "--------------------------------------------"
    python training_classification.py --config "$config"
    echo ""
done
