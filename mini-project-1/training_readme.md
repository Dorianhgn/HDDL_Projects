# Training Configuration

## Utilisation

```bash
python training_classification.py --config config.yaml
```

## Options YAML

### Entraînement
- **`epochs`** : nombre d'époques (défaut: `30`)
- **`batch_size`** : taille du batch (défaut: `32`)
- **`lr`** : learning rate (défaut: `1e-3`)
- **`weight_decay`** : régularisation AdamW (défaut: `1e-4`)

### Tâche & Données
- **`task`** : type de tâche — `"classification"`, `"contours"`, `"segmentation"`, `"box"` (défaut: `"contours"`)
- **`data_path`** : chemin vers le dataset (défaut: `"data/oxford-iiit-pet"`)

### Modèle
- **`model`** : chemin pointillé vers la classe du modèle, ex. `"models.unet.UNet"` (défaut: `"models.unet.UNet"`)
- **`model_args`** : arguments du constructeur (dict ou liste de dicts)
  ```yaml
  model_args:
    - n_s: 3
    - n_classes: 3
  ```

### Loss
- **`criterion`** : fonction de perte — `"CrossEntropyLoss"`, `"BCEWithLogitsLoss"`, `"MSELoss"` (défaut: `"CrossEntropyLoss"`)

### Sauvegarde
- **`path`** : dossier de sauvegarde des checkpoints (défaut: `"models/unet_ns3/"`)
- **`continue_training`** : reprendre depuis `last.pth` (flag booléen)

## Exemple de config.yaml

```yaml
epochs: 50
batch_size: 32
lr: 0.001
weight_decay: 0.0001
task: "classification"
criterion: "CrossEntropyLoss"
data_path: data/oxford-iiit-pet
path: models/unet_ns4/
model: models.unet.UNet
model_args:
  - n_s: 3
  - n_classes: 3
```

## Scheduler & Optimiseur

- **Optimiseur** : `AdamW` avec weight decay
- **Scheduler** : `CosineAnnealingLR` (T_max = epochs)
