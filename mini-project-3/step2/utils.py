"""
Utility functions for ViT vs CNN analysis on CLEVR reasoning tasks
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from captum.attr import LayerGradCam, IntegratedGradients, Occlusion
from captum.attr import visualization as viz

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from step2.models import get_resnet50_model, get_vit_b_16_model
from step2.dataloader import CLEVR_ReasoningDataset

import json


# --- GLOBAL CONSTANTS ---
LABEL_NAMES = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# --- DATASET & MODEL LOADING ---

def get_dataset(task, split='val'):
    """Load CLEVR dataset for a given task and split"""
    json_file = f"step2/task{task}_{split}.json"
    img_root_dir = f"step2/data_clevr/CLEVR_v1.0/images/{split}"
    return CLEVR_ReasoningDataset(json_file, img_root_dir, transform=TRANSFORM)


def load_models(task, device='cuda'):
    """
    Load ResNet50 and ViT-B/16 models for a given task
    
    Args:
        task: Task number (1, 2, 3, or 4)
        device: Device to load models on
        
    Returns:
        resnet, vit: Loaded models in eval mode
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Define checkpoint paths based on task
    if task == 1:
        ckpt_resnet = "step2/experiments/ResNet_task1/best_model.pth"
        ckpt_vit = "step2/experiments/ViT_task1/best_model.pth"
    elif task == 2:
        ckpt_resnet = "step2/experiments/ResNet_task2/best_model.pth"
        ckpt_vit = "step2/experiments/ViT_task2_bis/best_model.pth"
    elif task == 3:
        ckpt_resnet = "step2/experiments/ResNet_task3_wd/best_model.pth"
        ckpt_vit = "step2/experiments/ViT_task3_wd/best_model.pth"
    elif task == 4:
        ckpt_resnet = "step2/experiments/ResNet_task4_wd/best_model_acc.pth"
        ckpt_vit = "step2/experiments/ViT_task4_wd/best_model_acc.pth"
    else:
        raise ValueError(f"Invalid task: {task}. Must be 1, 2, 3, or 4")
    
    # Load ResNet
    resnet = get_resnet50_model(num_classes=8).to(device)
    resnet.load_state_dict(torch.load(ckpt_resnet, map_location=device))
    resnet.eval()
    
    # Load ViT
    vit = get_vit_b_16_model(num_classes=8).to(device)
    vit.load_state_dict(torch.load(ckpt_vit, map_location=device))
    vit.eval()
    
    return resnet, vit


def denormalize(img_tensor):
    """Reverse the ImageNet normalization for display"""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = img_tensor.numpy() * std + mean
    img = np.clip(img, 0, 1)
    return np.transpose(img, (1, 2, 0))


def get_sample_image(task, idx=0, split='val', device='cuda'):
    """
    Load a sample image and its label
    
    Returns:
        input_tensor: Normalized tensor (1, 3, 224, 224) on device
        label_idx: Ground truth label index
        original_img: Denormalized image for display (H, W, 3)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(task, split)
    img_tensor, label = dataset[idx]
    
    # Denormalize for display
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img_display = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
    img_display = np.clip(img_display, 0, 1)
    
    return img_tensor.unsqueeze(0).to(device), label, img_display


# --- HELPER FUNCTIONS FOR PYTORCH-GRAD-CAM ---

def reshape_transform_vit(tensor):
    """
    Fonction critique pour ViT : transforme la séquence 1D en image 2D.
    Basé sur le code source torchvision : output [Batch, SeqLen, Hidden]
    """
    # On enlève le token CLS (index 0) : [B, 197, 768] -> [B, 196, 768]
    height = 14
    width = 14
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    
    # Permutation pour passer de [B, H, W, C] à [B, C, H, W] (format image)
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# --- METRICS PLOTTING ---

def plot_training_metrics(task, save_path=None):
    """
    Plot training and validation losses/accuracies for ResNet and ViT
    
    Args:
        task: Task number (1, 2, 3, or 4)
        save_path: Optional path to save the figure
    """
    # Load metrics
    if task == 1:
        metrics_vit_path = "step2/experiments/ViT_task1/metrics.npy"
        metrics_resnet_path = "step2/experiments/ResNet_task1/metrics.npy"
    elif task == 2:
        metrics_vit_path = "step2/experiments/ViT_task2_bis/metrics.npy"
        metrics_resnet_path = "step2/experiments/ResNet_task2/metrics.npy"
    elif task == 3:
        metrics_vit_path = "step2/experiments/ViT_task3_wd/metrics.npy"
        metrics_resnet_path = "step2/experiments/ResNet_task3_wd/metrics.npy"
    elif task == 4:
        metrics_vit_path = "step2/experiments/ViT_task4_wd/metrics.npy"
        metrics_resnet_path = "step2/experiments/ResNet_task4_wd/metrics.npy"
    else:
        raise ValueError(f"Invalid task: {task}")
    
    metrics_vit = np.load(metrics_vit_path, allow_pickle=True).item()
    metrics_resnet = np.load(metrics_resnet_path, allow_pickle=True).item()
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # Training Loss
    axs[0, 0].plot(metrics_vit['train_loss'], '-o', label='ViT Train Loss')
    axs[0, 0].plot(metrics_resnet['train_loss'], '-s', label='ResNet Train Loss')
    axs[0, 0].set_title('Training Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(alpha=0.3)
    
    # Validation Loss
    axs[0, 1].plot(metrics_vit['val_loss'], '-o', label='ViT Val Loss')
    axs[0, 1].plot(metrics_resnet['val_loss'], '-s', label='ResNet Val Loss')
    axs[0, 1].set_title('Validation Loss')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    axs[0, 1].grid(alpha=0.3)
    
    # Training Accuracy
    axs[1, 0].plot(metrics_vit['train_acc'], '-o', label='ViT Train Acc')
    axs[1, 0].plot(metrics_resnet['train_acc'], '-s', label='ResNet Train Acc')
    axs[1, 0].set_title('Training Accuracy')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].legend()
    axs[1, 0].grid(alpha=0.3)
    
    # Validation Accuracy
    axs[1, 1].plot(metrics_vit['val_acc'], '-o', label='ViT Val Acc')
    axs[1, 1].plot(metrics_resnet['val_acc'], '-s', label='ResNet Val Acc')
    axs[1, 1].set_title('Validation Accuracy')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Accuracy')
    axs[1, 1].legend()
    axs[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# --- INTERPRETABILITY METHODS ---

def compute_gradcam_resnet(model, input_tensor, target_cls):
    """Utilise pytorch-grad-cam sur la dernière couche du ResNet"""
    # Cible : Dernière couche du dernier bloc bottleneck
    target_layers = [model.layer4[-1]]
    
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_cls)]
    
    # Génère la heatmap (valeurs entre 0 et 1)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    return grayscale_cam[0, :] # Retourne (224, 224)

def compute_gradcam_vit(model, input_tensor, target_cls):
    """Utilise pytorch-grad-cam avec reshape sur le ViT"""
    # Cible : La LayerNorm AVANT l'attention du dernier bloc (ln_1)
    # C'est là que les gradients spatiaux sont les plus riches
    target_layers = [model.encoder.layers[-1].ln_1]
    
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform_vit)
    targets = [ClassifierOutputTarget(target_cls)]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    return grayscale_cam[0, :] # Retourne (224, 224)


def compute_integrated_gradients(model, input_tensor, target_cls, n_steps=50):
    """
    Compute Integrated Gradients for pixel-level attributions
    
    Args:
        model: Model (ResNet or ViT)
        input_tensor: Input image tensor (1, 3, 224, 224)
        target_cls: Target class index
        n_steps: Number of integration steps
        
    Returns:
        Attribution map (1, 3, 224, 224)
    """
    ig = IntegratedGradients(model)
    attr = ig.attribute(input_tensor, target=target_cls, n_steps=n_steps)
    return attr

def compute_integrated_gradients_vit(model, input_tensor, target_cls, n_steps=50):
    """
    Compute Integrated Gradients for ViT model
    
    Args:
        model: ViT model
        input_tensor: Input image tensor (1, 3, 224, 224)
        target_cls: Target class index
        n_steps: Number of integration steps
    Returns:
        Attribution map (1, 3, 224, 224)
    """
    ig = IntegratedGradients(model)
    attr = ig.attribute(input_tensor, target=target_cls, n_steps=n_steps)
    
    # attr shape: (1, 3, 224, 224)
    # Take absolute values and average across color channels
    attr = torch.abs(attr)
    attr = attr.mean(dim=1, keepdim=True)  # (1, 1, 224, 224)
    
    return attr

def compute_occlusion(model, input_tensor, target_cls, window_size=32, stride=16):
    """
    Compute Occlusion-based attributions
    
    Args:
        model: Model (ResNet or ViT)
        input_tensor: Input image tensor (1, 3, 224, 224)
        target_cls: Target class index
        window_size: Size of occlusion window
        stride: Stride for sliding window
        
    Returns:
        Attribution map (1, 3, 224, 224)
    """
    occ = Occlusion(model)
    attr = occ.attribute(
        input_tensor, 
        target=target_cls,
        strides=(3, stride, stride),
        sliding_window_shapes=(3, window_size, window_size),
        baselines=0
    )
    return attr

def compute_occlusion_vit(model, input_tensor, target_cls, window_size=32, stride=16):
    """
    Compute Occlusion-based attributions for ViT model
    
    Args:
        model: ViT model
        input_tensor: Input image tensor (1, 3, 224, 224)
        target_cls: Target class index
        window_size: Size of occlusion window
        stride: Stride for sliding window
    Returns:
        Attribution map (1, 3, 224, 224)
    """
    occ = Occlusion(model)
    attr = occ.attribute(
        input_tensor, 
        target=target_cls,
        strides=(3, stride, stride),
        sliding_window_shapes=(3, window_size, window_size),
        baselines=0
    )

    return attr


# --- VISUALIZATION FUNCTIONS ---

def plot_gradcam_comparison(resnet, vit, input_tensor, target_cls, original_img, 
                           title_prefix="", save_path=None):
    """Affiche le GradCAM généré par la nouvelle librairie"""
    
    # 1. Compute (Retourne des numpy array 2D normalisés 0-1)
    cam_resnet = compute_gradcam_resnet(resnet, input_tensor, target_cls)
    cam_vit = compute_gradcam_vit(vit, input_tensor, target_cls)
    
    # 2. Overlay sur l'image originale pour le style "Jet"
    vis_resnet = show_cam_on_image(original_img, cam_resnet, use_rgb=True)
    vis_vit = show_cam_on_image(original_img, cam_vit, use_rgb=True)
    
    # 3. Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_img)
    axes[0].set_title(f"{title_prefix}Original\nTrue: {LABEL_NAMES[target_cls]}", fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(vis_resnet)
    axes[1].set_title(f"{title_prefix}ResNet GradCAM", fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(vis_vit)
    axes[2].set_title(f"{title_prefix}ViT GradCAM (Reshaped)", fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.show()


def plot_integrated_gradients_comparison(resnet, vit, input_tensor, target_cls, original_img,
                                        n_steps=50, title_prefix="", save_path=None):
    """
    Plot side-by-side Integrated Gradients comparison
    
    Args:
        resnet: ResNet model
        vit: ViT model
        input_tensor: Input image tensor
        target_cls: Target class index
        original_img: Original image for visualization
        n_steps: Number of integration steps
        title_prefix: Prefix for titles
        save_path: Optional save path
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title(f"{title_prefix}Original\nTrue Label: {LABEL_NAMES[target_cls]}", fontsize=12)
    axes[0].axis('off')
    
    # ResNet Integrated Gradients
    attr_resnet = compute_integrated_gradients(resnet, input_tensor, target_cls, n_steps)
    attr_resnet_np = attr_resnet.squeeze(0).cpu().detach().numpy()
    attr_resnet_np = np.transpose(attr_resnet_np, (1, 2, 0))
    
    # CORRECTION CRITIQUE : Pour appliquer une cmap 'Blues', il faut une map 2D (1 canal), pas 3D (RGB).
    # On prend la moyenne des valeurs absolues sur les canaux couleurs.
    heatmap_resnet = np.mean(np.abs(attr_resnet_np), axis=2)
    
    # Astuce visuelle : On n'affiche pas l'image originale en fond pour le ResNet IG "Pixel Level".
    # On affiche juste le bruit bleu sur fond blanc, c'est beaucoup plus lisible pour voir les "bords" détectés.
    # Si tu veux voir l'objet en dessous, décommente la ligne suivante avec alpha très faible.
    # axes[1].imshow(original_img, alpha=0.15) 
    
    # Affichage du bruit bleu
    im1 = axes[1].imshow(heatmap_resnet, cmap='Blues', alpha=1.0, vmin=0, vmax=np.max(heatmap_resnet))
    axes[1].set_title(f"{title_prefix}ResNet IG (Pixel Focus)", fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # ViT Integrated Gradients
    attr_vit = compute_integrated_gradients_vit(vit, input_tensor, target_cls, n_steps)
    attr_vit_np = attr_vit.squeeze(0).cpu().detach().numpy()
    attr_vit_np = np.transpose(attr_vit_np, (1, 2, 0))
    
    # Même chose : on écrase les canaux pour avoir une intensité globale
    heatmap_vit = np.mean(np.abs(attr_vit_np), axis=2)
    
    # Ici on veut voir où ça tombe sur l'objet -> On affiche l'original en fond
    axes[2].imshow(original_img, alpha=0.6) # Image de fond plus visible
    im2 = axes[2].imshow(heatmap_vit, cmap='jet', alpha=0.5, vmin=0, vmax=np.max(heatmap_vit))
    axes[2].set_title(f"{title_prefix}ViT IG (Attention Focus)", fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_occlusion_comparison(resnet, vit, input_tensor, target_cls, original_img,
                              window_size=32, stride=16, title_prefix="", save_path=None):
    """
    Plot side-by-side Occlusion comparison
    
    Args:
        resnet: ResNet model
        vit: ViT model
        input_tensor: Input image tensor
        target_cls: Target class index
        original_img: Original image for visualization
        window_size: Occlusion window size
        stride: Stride for sliding window
        title_prefix: Prefix for titles
        save_path: Optional save path
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title(f"{title_prefix}Original\nTrue Label: {LABEL_NAMES[target_cls]}", fontsize=12)
    axes[0].axis('off')
    
    # ResNet Occlusion
    attr_resnet = compute_occlusion(resnet, input_tensor, target_cls, window_size, stride)
    attr_resnet_np = attr_resnet.squeeze(0).cpu().detach().numpy()
    attr_resnet_np = np.transpose(attr_resnet_np, (1, 2, 0))
    
    # Transformation en Heatmap 2D (Moyenne des valeurs absolues sur RGB)
    heatmap_resnet = np.mean(np.abs(attr_resnet_np), axis=2)
    
    # Affichage
    axes[1].imshow(original_img, alpha=0.6) # Image de fond
    im1 = axes[1].imshow(heatmap_resnet, cmap='jet', alpha=0.5, vmin=0, vmax=np.max(heatmap_resnet))
    axes[1].set_title(f"{title_prefix}ResNet Occlusion", fontsize=12)
    axes[1].axis('off')
    
    # ViT Occlusion
    attr_vit = compute_occlusion_vit(vit, input_tensor, target_cls, window_size, stride)
    # Moyenne sur les canaux RGB pour l'affichage
    attr_vit_np = attr_vit.squeeze(0).cpu().detach().numpy()
    attr_vit_np = np.transpose(attr_vit_np, (1, 2, 0))
    
# Transformation en Heatmap 2D
    heatmap_vit = np.mean(np.abs(attr_vit_np), axis=2)
    
    # Affichage
    axes[2].imshow(original_img, alpha=0.6) # Image de fond
    im2 = axes[2].imshow(heatmap_vit, cmap='jet', alpha=0.5, vmin=0, vmax=np.max(heatmap_vit))
    axes[2].set_title(f"{title_prefix}ViT Occlusion", fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# --- ERROR ANALYSIS ---

def find_prediction_cases(resnet, vit, task, split='val', device='cuda'):
    """
    Find interesting prediction cases: CNN wrong/ViT right, vice versa, both wrong
    
    Returns:
        Dictionary with lists of indices for each case:
        - 'cnn_wrong_vit_right': indices where CNN fails but ViT succeeds
        - 'vit_wrong_cnn_right': indices where ViT fails but CNN succeeds
        - 'both_wrong': indices where both models fail
        - 'both_right': indices where both models succeed
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(task, split)
    
    cases = {
        'cnn_wrong_vit_right': [],
        'vit_wrong_cnn_right': [],
        'both_wrong': [],
        'both_right': []
    }
    
    resnet.eval()
    vit.eval()
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            img_tensor, label = dataset[idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            # Get predictions
            resnet_pred = resnet(img_tensor).argmax(dim=1).item()
            vit_pred = vit(img_tensor).argmax(dim=1).item()
            label = label.item()
            
            # Categorize
            resnet_correct = (resnet_pred == label)
            vit_correct = (vit_pred == label)
            
            if resnet_correct and vit_correct:
                cases['both_right'].append(idx)
            elif not resnet_correct and vit_correct:
                cases['cnn_wrong_vit_right'].append(idx)
            elif resnet_correct and not vit_correct:
                cases['vit_wrong_cnn_right'].append(idx)
            else:
                cases['both_wrong'].append(idx)
    
    # Print summary
    print(f"Task {task} - {split.upper()} set analysis:")
    print(f"  Both correct: {len(cases['both_right'])} ({100*len(cases['both_right'])/len(dataset):.1f}%)")
    print(f"  CNN wrong, ViT right: {len(cases['cnn_wrong_vit_right'])} ({100*len(cases['cnn_wrong_vit_right'])/len(dataset):.1f}%)")
    print(f"  ViT wrong, CNN right: {len(cases['vit_wrong_cnn_right'])} ({100*len(cases['vit_wrong_cnn_right'])/len(dataset):.1f}%)")
    print(f"  Both wrong: {len(cases['both_wrong'])} ({100*len(cases['both_wrong'])/len(dataset):.1f}%)")
    
    return cases


def analyze_sample(resnet, vit, task, idx, split='val', device='cuda', methods=['gradcam', 'ig', 'occlusion']):
    """
    Analyze predictions for a single sample using multiple interpretability methods
    
    Args:
        resnet: ResNet model
        vit: ViT model
        task: Task number (2, 3, or 4)
        idx: Index of sample in dataset
        split: Dataset split ('val' or 'train')
        device: Device to run on
        methods: List of methods to use ['gradcam', 'ig', 'occlusion']
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Get sample
    input_tensor, label_idx, original_img = get_sample_image(task, idx, split, device)
    
    # Get predictions
    with torch.no_grad():
        resnet_out = resnet(input_tensor)
        vit_out = vit(input_tensor)
        resnet_pred = resnet_out.argmax(dim=1).item()
        vit_pred = vit_out.argmax(dim=1).item()
    
    # Print predictions
    print(f"Ground Truth: {LABEL_NAMES[label_idx]}")
    print(f"ResNet Prediction: {LABEL_NAMES[resnet_pred]} {'✓' if resnet_pred == label_idx else '✗'}")
    print(f"ViT Prediction: {LABEL_NAMES[vit_pred]} {'✓' if vit_pred == label_idx else '✗'}")
    print()
    
    # Apply interpretability methods
    if 'gradcam' in methods:
        plot_gradcam_comparison(resnet, vit, input_tensor, label_idx, original_img)
    
    if 'ig' in methods:
        plot_integrated_gradients_comparison(resnet, vit, input_tensor, label_idx, original_img)
    
    if 'occlusion' in methods:
        plot_occlusion_comparison(resnet, vit, input_tensor, label_idx, original_img)


# --- ATTRIBUTE-BASED ERROR ANALYSIS ---

def load_metadata(task, split='val'):
    """Load metadata for a given task and split"""
    metadata_file = f"step2/task{task}_{split}_metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    return metadata


def analyze_errors_by_attribute(model, task, split='val', device='cuda', model_name='Model'):
    """
    Analyze model errors by object attributes (color, material, shape, size)
    
    Args:
        model: The model to analyze (ResNet or ViT)
        task: Task number (2, 3, or 4)
        split: Dataset split ('val' or 'train')
        device: Device to run on
        model_name: Name of the model for display
        
    Returns:
        Dictionary with error statistics by attribute
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Load dataset and metadata
    dataset = get_dataset(task, split)
    metadata = load_metadata(task, split)
    
    # Initialize counters
    error_stats = {
        'anchor': {'color': {}, 'material': {}, 'shape': {}, 'size': {}},
        'target': {'color': {}, 'material': {}, 'shape': {}, 'size': {}},
    }
    
    # Analyze each sample
    correct_count = 0
    total_count = 0
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            img_tensor, label = dataset[idx]
            input_tensor = img_tensor.unsqueeze(0).to(device)
            
            # Get prediction
            output = model(input_tensor)
            pred = output.argmax(dim=1).item()
            
            # Check if correct
            is_correct = (pred == label)
            total_count += 1
            if is_correct:
                correct_count += 1
                continue
            
            # If wrong, analyze attributes
            meta = metadata[idx]
            
            # Count errors by anchor attributes
            for attr in ['color', 'material', 'shape', 'size']:
                anchor_val = meta['anchor'][attr]
                if anchor_val not in error_stats['anchor'][attr]:
                    error_stats['anchor'][attr][anchor_val] = {'errors': 0, 'total': 0}
                error_stats['anchor'][attr][anchor_val]['errors'] += 1
            
            # Count errors by target attributes
            for attr in ['color', 'material', 'shape', 'size']:
                target_val = meta['target'][attr]
                if target_val not in error_stats['target'][attr]:
                    error_stats['target'][attr][target_val] = {'errors': 0, 'total': 0}
                error_stats['target'][attr][target_val]['errors'] += 1
    
    # Count totals for all samples
    for idx in range(len(dataset)):
        meta = metadata[idx]
        
        for attr in ['color', 'material', 'shape', 'size']:
            anchor_val = meta['anchor'][attr]
            if anchor_val not in error_stats['anchor'][attr]:
                error_stats['anchor'][attr][anchor_val] = {'errors': 0, 'total': 0}
            error_stats['anchor'][attr][anchor_val]['total'] += 1
            
            target_val = meta['target'][attr]
            if target_val not in error_stats['target'][attr]:
                error_stats['target'][attr][target_val] = {'errors': 0, 'total': 0}
            error_stats['target'][attr][target_val]['total'] += 1
    
    # Calculate error rates
    for obj_type in ['anchor', 'target']:
        for attr in ['color', 'material', 'shape', 'size']:
            for val, counts in error_stats[obj_type][attr].items():
                if counts['total'] > 0:
                    counts['error_rate'] = counts['errors'] / counts['total']
                else:
                    counts['error_rate'] = 0.0
    
    accuracy = correct_count / total_count
    print(f"\n{model_name} - Task {task} - {split.upper()}")
    print(f"Overall Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    print(f"Total Errors: {total_count - correct_count}")
    
    return error_stats


def plot_error_analysis(error_stats, task, model_name='Model', save_path=None):
    """
    Plot error analysis by attributes
    
    Args:
        error_stats: Dictionary returned by analyze_errors_by_attribute
        task: Task number
        model_name: Name of the model for display
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'{model_name} - Task {task} - Error Analysis by Attribute', 
                 fontsize=16, fontweight='bold')
    
    obj_types = ['anchor', 'target']
    obj_labels = ['Object to Detect (Anchor)', 'Object to Predict (Target)']
    attributes = ['color', 'material', 'shape', 'size']
    
    for row, (obj_type, obj_label) in enumerate(zip(obj_types, obj_labels)):
        for col, attr in enumerate(attributes):
            ax = axes[row, col]
            
            data = error_stats[obj_type][attr]
            
            if not data:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f'{obj_label}\n{attr.capitalize()}')
                continue
            
            # Prepare data for plotting
            values = list(data.keys())
            error_rates = [data[v]['error_rate'] * 100 for v in values]
            totals = [data[v]['total'] for v in values]
            
            # Sort by error rate
            sorted_indices = sorted(range(len(error_rates)), key=lambda i: error_rates[i], reverse=True)
            values = [values[i] for i in sorted_indices]
            error_rates = [error_rates[i] for i in sorted_indices]
            totals = [totals[i] for i in sorted_indices]
            
            # Create bar plot
            bars = ax.barh(values, error_rates, color='crimson', alpha=0.7)
            
            # Add sample count labels
            for i, (bar, total) in enumerate(zip(bars, totals)):
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                       f'n={total}', va='center', fontsize=9)
            
            ax.set_xlabel('Error Rate (%)', fontsize=10)
            ax.set_title(f'{obj_label}\n{attr.capitalize()}', fontsize=11, fontweight='bold')
            ax.set_xlim(0, max(error_rates) * 1.2 if error_rates else 100)
            ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_models_by_attribute(resnet, vit, task, split='val', device='cuda', save_path=None):
    """
    Compare ResNet and ViT error patterns by attributes
    
    Args:
        resnet: ResNet model
        vit: ViT model
        task: Task number (2, 3, or 4)
        split: Dataset split ('val' or 'train')
        device: Device to run on
        save_path: Optional path to save the figure
    """
    print("Analyzing ResNet errors...")
    resnet_stats = analyze_errors_by_attribute(resnet, task, split, device, 'ResNet')
    
    print("\nAnalyzing ViT errors...")
    vit_stats = analyze_errors_by_attribute(vit, task, split, device, 'ViT')
    
    # Plot comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Task {task} - Error Comparison: ResNet vs ViT', 
                 fontsize=16, fontweight='bold')
    
    obj_types = ['anchor', 'target']
    obj_labels = ['Object to Detect (Anchor)', 'Object to Predict (Target)']
    attributes = ['color', 'material', 'shape', 'size']
    
    for row, (obj_type, obj_label) in enumerate(zip(obj_types, obj_labels)):
        for col, attr in enumerate(attributes):
            ax = axes[row, col]
            
            resnet_data = resnet_stats[obj_type][attr]
            vit_data = vit_stats[obj_type][attr]
            
            # Get all unique values
            all_values = set(resnet_data.keys()) | set(vit_data.keys())
            
            if not all_values:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f'{obj_label}\n{attr.capitalize()}')
                continue
            
            # Prepare data
            values = sorted(all_values)
            resnet_rates = [resnet_data.get(v, {}).get('error_rate', 0) * 100 for v in values]
            vit_rates = [vit_data.get(v, {}).get('error_rate', 0) * 100 for v in values]
            
            # Create grouped bar plot
            x = np.arange(len(values))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, resnet_rates, width, label='ResNet', 
                          color='steelblue', alpha=0.8)
            bars2 = ax.bar(x + width/2, vit_rates, width, label='ViT', 
                          color='darkorange', alpha=0.8)
            
            ax.set_xlabel(attr.capitalize(), fontsize=10)
            ax.set_ylabel('Error Rate (%)', fontsize=10)
            ax.set_title(f'{obj_label}\n{attr.capitalize()}', fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(values, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return resnet_stats, vit_stats


def print_attribute_summary(error_stats, model_name='Model'):
    """
    Print a summary of error patterns by attribute
    
    Args:
        error_stats: Dictionary returned by analyze_errors_by_attribute
        model_name: Name of the model for display
    """
    print(f"\n{'='*80}")
    print(f"{model_name} - Attribute Error Summary")
    print(f"{'='*80}\n")
    
    obj_types = ['anchor', 'target']
    obj_labels = ['Object to Detect (Anchor)', 'Object to Predict (Target)']
    attributes = ['color', 'material', 'shape', 'size']
    
    for obj_type, obj_label in zip(obj_types, obj_labels):
        print(f"\n{obj_label}:")
        print("-" * 80)
        
        for attr in attributes:
            data = error_stats[obj_type][attr]
            
            if not data:
                continue
            
            # Find attribute value with highest error rate
            max_error_val = max(data.items(), key=lambda x: x[1]['error_rate'])
            min_error_val = min(data.items(), key=lambda x: x[1]['error_rate'])
            
            print(f"\n  {attr.upper()}:")
            print(f"    Highest error rate: {max_error_val[0]} "
                  f"({max_error_val[1]['error_rate']:.1%}, "
                  f"{max_error_val[1]['errors']}/{max_error_val[1]['total']})")
            print(f"    Lowest error rate:  {min_error_val[0]} "
                  f"({min_error_val[1]['error_rate']:.1%}, "
                  f"{min_error_val[1]['errors']}/{min_error_val[1]['total']})")
    
    print(f"\n{'='*80}\n")
