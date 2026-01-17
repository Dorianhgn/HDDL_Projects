# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# # Task i

# %% [markdown]
# ## Display examples

# %%
from step2.dataloader import CLEVR_ReasoningDataset
from torchvision import transforms

TASK = 4
IDX_IMG = 5  # Choisir l'index de l'image à analyser

# 1. Définir les transformations (Vital pour ViT : resize 224x224 + Norm ImageNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)), # ViT a besoin de patchs fixes
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if TASK == 1: # La couleur du cylindre le plus loin
    test_dataset = CLEVR_ReasoningDataset(
        json_file="step2/task1_val.json",
        img_root_dir="step2/data_clevr/CLEVR_v1.0/images/val",
        transform=transform
    )
elif TASK == 2: # La couleur de l'objet le plus loing de la sphère grise
    # 2. Instancier le Dataset pour la Tâche 2 (Sphère Grise)
    test_dataset = CLEVR_ReasoningDataset(
        json_file="step2/task2_val.json",
        img_root_dir="step2/data_clevr/CLEVR_v1.0/images/val",
        transform=transform
    )
elif TASK == 3:
    test_dataset = CLEVR_ReasoningDataset(
        json_file="step2/task3_val.json",
        img_root_dir="step2/data_clevr/CLEVR_v1.0/images/val",
        transform=transform
    )
elif TASK == 4:
    test_dataset = CLEVR_ReasoningDataset(
        json_file="step2/task4_val.json",
        img_root_dir="step2/data_clevr/CLEVR_v1.0/images/val",
        transform=transform
    )

# %%
# Display several image examples and the answer (can be one label or two later on)

# Define label names (8 colors)
LABEL_NAMES = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']

# Function to denormalize images
def denormalize(img_tensor):
    """Reverse the ImageNet normalization"""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = img_tensor.numpy() * std + mean
    img = np.clip(img, 0, 1)
    return np.transpose(img, (1, 2, 0))

# Display 8 random examples
num_examples = 8
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

# Get random indices
random_indices = np.random.choice(len(test_dataset), num_examples, replace=False)

for idx, ax in enumerate(axes):
    img, label = test_dataset[random_indices[idx]]
    img_display = denormalize(img)
    
    ax.imshow(img_display)
    label_name = LABEL_NAMES[label.item()]
    ax.set_title(f"Label: {label_name} ({label.item()})", fontsize=12, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Check losses convergence ResNet vs ViT

# %%
if TASK == 1:
    METRICS_VIT_PATH = "step2/experiments/ViT_task1/metrics.npy"
    METRICS_RESNET_PATH = "step2/experiments/ResNet_task1/metrics.npy"
elif TASK == 2: 
    METRICS_VIT_PATH = "step2/experiments/ViT_task2_bis/metrics.npy"
    METRICS_RESNET_PATH = "step2/experiments/ResNet_task2/metrics.npy"
elif TASK == 3:
    METRICS_VIT_PATH = "step2/experiments/ViT_task3_wd/metrics.npy"
    METRICS_RESNET_PATH = "step2/experiments/ResNet_task3_wd/metrics.npy"
elif TASK == 4:
    METRICS_VIT_PATH = "step2/experiments/ViT_task4_wd/metrics.npy"
    METRICS_RESNET_PATH = "step2/experiments/ResNet_task4_wd/metrics.npy"

# %%
metrics_vit = np.load(METRICS_VIT_PATH, allow_pickle=True).item()
metrics_resnet = np.load(METRICS_RESNET_PATH, allow_pickle=True).item()

# %%
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot training, then validation loss, then train accuracy, then val accuracy
axs[0, 0].plot(metrics_vit['train_loss'], '-o', label='ViT Train Loss')
axs[0, 0].plot(metrics_resnet['train_loss'], '-s', label='ResNet Train Loss')
axs[0, 0].set_title('Training Loss')
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].legend()

axs[0, 1].plot(metrics_vit['val_loss'],'-o',  label='ViT Val Loss')
axs[0, 1].plot(metrics_resnet['val_loss'], '-s', label='ResNet Val Loss')
axs[0, 1].set_title('Validation Loss')
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend()

axs[1, 0].plot(metrics_vit['train_acc'],'-o',  label='ViT Train Acc')
axs[1, 0].plot(metrics_resnet['train_acc'], '-s', label='ResNet Train Acc')
axs[1, 0].set_title('Training Accuracy')
axs[1, 0].set_xlabel('Epochs')
axs[1, 0].set_ylabel('Accuracy')
axs[1, 0].legend()

axs[1, 1].plot(metrics_vit['val_acc'], '-o', label='ViT Val Acc')
axs[1, 1].plot(metrics_resnet['val_acc'], '-s', label='ResNet Val Acc')
axs[1, 1].set_title('Validation Accuracy')
axs[1, 1].set_xlabel('Epochs')
axs[1, 1].set_ylabel('Accuracy')
axs[1, 1].legend()

plt.tight_layout()
plt.show()

# %%
# !pip install captum > /dev/null

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# --- Imports Captum ---
try:
    from captum.attr import LayerGradCam, Occlusion, IntegratedGradients
    from captum.attr import visualization as viz
except ImportError:
    import sys
    !{sys.executable} -m pip install captum

    from captum.attr import LayerGradCam, Occlusion, IntegratedGradients
    from captum.attr import visualization as viz

# --- Tes imports locaux ---
from step2.models import get_resnet50_model, get_vit_b_16_model
from step2.dataloader import CLEVR_ReasoningDataset

# --- 1. SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if TASK == 1:
    CKPT_RESNET = "step2/experiments/ResNet_task1/best_model.pth"
    CKPT_VIT = "step2/experiments/ViT_task1/best_model.pth"
elif TASK == 2:
    CKPT_RESNET = "step2/experiments/ResNet_task2/best_model.pth"
    CKPT_VIT = "step2/experiments/ViT_task2_bis/best_model.pth"
elif TASK == 3:
    CKPT_RESNET = "step2/experiments/ResNet_task3_wd/best_model.pth"
    CKPT_VIT = "step2/experiments/ViT_task3_wd/best_model.pth"
elif TASK == 4:
    CKPT_RESNET = "step2/experiments/ResNet_task4_wd/best_model_acc.pth"
    CKPT_VIT = "step2/experiments/ViT_task4_wd/best_model_acc.pth"

# Transformations (Identiques au training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Fonction pour charger une image et son label
def get_sample_image(json_file, img_dir, idx=0):
    dataset = CLEVR_ReasoningDataset(json_file, img_dir, transform=transform)
    img_tensor, label = dataset[idx]
    # Pour l'affichage (dé-normalisation)
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img_display = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
    img_display = np.clip(img_display, 0, 1)
    return img_tensor.unsqueeze(0).to(device), label, img_display

# --- 2. CHARGEMENT DES MODÈLES ---
# ResNet
resnet = get_resnet50_model(num_classes=8).to(device)
resnet.load_state_dict(torch.load(CKPT_RESNET, map_location=device))
resnet.eval()

# ViT
vit = get_vit_b_16_model(num_classes=8).to(device)
vit.load_state_dict(torch.load(CKPT_VIT, map_location=device))
vit.eval()

# Récupération d'une image (Change idx pour explorer !)
input_tensor, label_idx, original_img = get_sample_image(
    f"step2/task{TASK}_val.json", 
    "step2/data_clevr/CLEVR_v1.0/images/val", 
    idx=IDX_IMG
)
print(f"True Label Index: {label_idx}")

# %%
# --- CELLULE ANALYSE RESNET ---
import torch
import numpy as np
from captum.attr import LayerGradCam, IntegratedGradients
from captum.attr import visualization as viz

# 0. Plot the image in color
plt.imshow(original_img)
plt.axis('off')
plt.title(f"Input Image (True Label: {label_idx})")
plt.show()

# 1. Setup Rapide
model = resnet # Assure-toi que 'resnet' est chargé et en .eval()
target_layer = model.layer4[-1] # Dernière couche de convolution
input_img = input_tensor # Ton image chargée précédemment
target_cls = label_idx   # La classe cible

# 2. Fonction d'affichage propre
def show_cam(attr, title):
    # Convertit (1, H, W) -> (H, W, 1) pour l'affichage
    attr = attr.squeeze(0).cpu().detach().numpy()
    attr = np.transpose(attr, (1, 2, 0))
    
    viz.visualize_image_attr(
        attr, original_img, method="blended_heat_map",
        sign="positive", show_colorbar=True, title=title
    )

print(f"--- Analyse ResNet50 (Classe: {target_cls}) ---")

# --- A. GRAD-CAM ---
# On regarde les activation maps finales
lgc = LayerGradCam(model, target_layer)
attr_gc = lgc.attribute(input_img, target=target_cls)
# Upsampling vers 224x224
attr_gc = LayerGradCam.interpolate(attr_gc, (224, 224))
show_cam(attr_gc, "ResNet - GradCAM (Texture Bias)")

# --- B. INTEGRATED GRADIENTS (Pixel-wise) ---
# Montre quels pixels précis ont contribué
ig = IntegratedGradients(model)
attr_ig = ig.attribute(input_img, target=target_cls, n_steps=20) # n_steps=20 pour aller vite
attr_ig_np = np.transpose(attr_ig.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

viz.visualize_image_attr(
    attr_ig_np, original_img, method="heat_map",
    sign="absolute_value", show_colorbar=True, title="ResNet - Pixel Importance"
)

# %%
# --- CELLULE ANALYSE VIT ---
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients, Occlusion

# 1. Setup
model = vit # Assure-toi que 'vit' est chargé et en .eval()
input_img = input_tensor
target_cls = label_idx

print(f"--- Analyse ViT-B/16 (Classe: {target_cls}) ---")

# --- A. INTEGRATED GRADIENTS (Pixel-wise) ---
# Pour les ViT, IntegratedGradients au niveau pixel est plus informatif car:
# - Le CLS token agrège toute l'information pour la classification
# - Les attributions des autres tokens sont souvent nulles
# - On veut voir quels PIXELS (pas patches) sont importants

ig = IntegratedGradients(model)
print("Calcul des Integrated Gradients (ça peut prendre 10-20s)...")
attr_ig = ig.attribute(input_img, target=target_cls, n_steps=50)

# Moyenne sur les canaux RGB pour l'affichage
attr_ig_np = attr_ig.squeeze(0).cpu().detach().numpy()
attr_ig_np = np.transpose(attr_ig_np, (1, 2, 0))

# Visualisation
viz.visualize_image_attr(
    attr_ig_np, original_img, method="blended_heat_map",
    sign="all", show_colorbar=True, title="ViT - Pixel Importance (Integrated Gradients)"
)

# --- B. OCCLUSION (Robustesse) ---
# "Si je cache cette zone, est-ce que tu perds la boule ?"
# Sliding window de 32x32 pixels (stride 16)
occ = Occlusion(model)
print("Calcul de l'occlusion en cours (ça peut prendre 30-60s)...")
attr_occ = occ.attribute(input_img, target=target_cls, 
                         strides=(3, 16, 16), 
                         sliding_window_shapes=(3, 32, 32), 
                         baselines=0)

# Moyenne sur les canaux RGB pour l'affichage
attr_occ_np = attr_occ.squeeze(0).cpu().detach().numpy()
attr_occ_np = np.transpose(attr_occ_np, (1, 2, 0))

viz.visualize_image_attr(
    attr_occ_np, original_img, method="blended_heat_map",
    sign="all", show_colorbar=True, title="ViT - Sensibilité Occlusion"
)


