"""
Utility functions for ConvNeXt vs Swin analysis on ImageNet-R-mini

Adapted for comparing CNN (ConvNeXt) vs Transformer (Swin) on image classification
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from captum.attr import IntegratedGradients, Occlusion

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from step4.models import get_convnext_model, get_swin_model
from step4.dataloader import get_test_loader

# --- GLOBAL CONSTANTS ---
CLASS_NAMES = ['ant', 'boxer', 'cannon', 'carousel', 'dalmatian', 'electric_guitar', 'french_bulldog', 'golden_retriever', 'goose', 'hotdog', 'jellyfish', 'ladybug', 'lion', 'lipstick', 'meerkat', 'missile', 'school_bus', 'spider_web', 'tank', 'toucan', 'vase']

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- MODEL LOADING ---

def load_models(device='cuda'):
    """
    Load ConvNeXt and Swin models
    
    Args:
        device: Device to load models on
        
    Returns:
        convnext, swin: Loaded models in eval mode
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Define checkpoint paths
    ckpt_convnext = "step4/checkpoints/convnext/convnext_best.pth"
    ckpt_swin = "step4/checkpoints/swin_lr0001_bs64/swin_best.pth"
    
    # Load ConvNeXt
    convnext = get_convnext_model(num_classes=21).to(device)
    convnext.load_state_dict(torch.load(ckpt_convnext, map_location=device)['model_state_dict'])
    convnext.eval()
    
    # Load Swin
    swin = get_swin_model(num_classes=21).to(device)
    swin.load_state_dict(torch.load(ckpt_swin, map_location=device)['model_state_dict'])
    swin.eval()
    
    return convnext, swin

def denormalize(img_tensor):
    """Reverse the ImageNet normalization for display"""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = img_tensor.numpy() * std + mean
    img = np.clip(img, 0, 1)
    return np.transpose(img, (1, 2, 0))

def get_sample_from_dataloader(dataloader, idx=0, device='cuda'):
    """
    Load a sample image from the dataloader
    
    Returns:
        input_tensor: Normalized tensor (1, 3, 224, 224) on device
        label_idx: Ground truth label index
        original_img: Denormalized image for display (H, W, 3)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Get the specific sample from dataset
    dataset = dataloader.dataset
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

def reshape_transform_swin(tensor):
    # Le Swin de torchvision sort souvent des tenseurs (B, H, W, C)
    # GradCAM a besoin de (B, C, H, W)
    result = tensor.permute(0, 3, 1, 2) 
    return result

# --- INTERPRETABILITY METHODS ---

def compute_gradcam_convnext(model, input_tensor, target_cls):
    """
    Utilise pytorch-grad-cam sur la dernière couche du ConvNeXt
    
    Pour ConvNeXt, la structure est: model.features[-1][-1]
    """
    # Cible : Dernière couche des features
    target_layers = [model.features[-1][-1]]
    
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_cls)]
    
    # Génère la heatmap (valeurs entre 0 et 1)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    return grayscale_cam[0, :]  # Retourne (224, 224)

def compute_gradcam_swin(model, input_tensor, target_cls):
    """
    Utilise pytorch-grad-cam avec reshape sur le Swin Transformer
    
    Pour Swin-T: model.layers[-1].blocks[-1].norm1
    """
    # Cible : LayerNorm du dernier bloc du dernier stage
    target_layers = [model.features[-1][-1].norm1]
    
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform_swin)
    targets = [ClassifierOutputTarget(target_cls)]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    return grayscale_cam[0, :]  # Retourne (224, 224)

def compute_integrated_gradients(model, input_tensor, target_cls, n_steps=50):
    """
    Compute Integrated Gradients for pixel-level attributions
    
    Args:
        model: Model (ConvNeXt or Swin)
        input_tensor: Input image tensor (1, 3, 224, 224)
        target_cls: Target class index
        n_steps: Number of integration steps
        
    Returns:
        Attribution map (1, 3, 224, 224)
    """
    ig = IntegratedGradients(model)
    attr = ig.attribute(input_tensor, target=target_cls, n_steps=n_steps)
    return attr

def compute_occlusion(model, input_tensor, target_cls, window_size=32, stride=16):
    """
    Compute Occlusion-based attributions
    
    Args:
        model: Model (ConvNeXt or Swin)
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

def plot_gradcam_comparison(convnext, swin, input_tensor, target_cls, original_img, 
                           title_prefix="", save_path=None):
    """Affiche le GradCAM généré pour ConvNeXt et Swin"""
    
    # 1. Compute (Retourne des numpy array 2D normalisés 0-1)
    cam_convnext = compute_gradcam_convnext(convnext, input_tensor, target_cls)
    cam_swin = compute_gradcam_swin(swin, input_tensor, target_cls)
    
    # 2. Overlay sur l'image originale pour le style "Jet"
    vis_convnext = show_cam_on_image(original_img, cam_convnext, use_rgb=True)
    vis_swin = show_cam_on_image(original_img, cam_swin, use_rgb=True)
    
    # 3. Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_img)
    axes[0].set_title(f"{title_prefix}Original\nTrue: {CLASS_NAMES[target_cls]}", fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(vis_convnext)
    axes[1].set_title(f"{title_prefix}ConvNeXt GradCAM", fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(vis_swin)
    axes[2].set_title(f"{title_prefix}Swin GradCAM", fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_integrated_gradients_comparison(convnext, swin, input_tensor, target_cls, original_img,
                                        n_steps=50, title_prefix="", save_path=None):
    """
    Plot side-by-side Integrated Gradients comparison
    
    Args:
        convnext: ConvNeXt model
        swin: Swin model
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
    axes[0].set_title(f"{title_prefix}Original\nTrue Label: {CLASS_NAMES[target_cls]}", fontsize=12)
    axes[0].axis('off')
    
    # ConvNeXt Integrated Gradients
    attr_convnext = compute_integrated_gradients(convnext, input_tensor, target_cls, n_steps)
    attr_convnext_np = attr_convnext.squeeze(0).cpu().detach().numpy()
    attr_convnext_np = np.transpose(attr_convnext_np, (1, 2, 0))
    
    # Pour afficher une heatmap 2D, on prend la moyenne des canaux
    heatmap_convnext = np.mean(np.abs(attr_convnext_np), axis=2)
    
    # Affichage du bruit bleu pour ConvNeXt (focus pixel-level)
    im1 = axes[1].imshow(heatmap_convnext, cmap='Blues', alpha=1.0, vmin=0, vmax=np.max(heatmap_convnext))
    axes[1].set_title(f"{title_prefix}ConvNeXt IG (Pixel Focus)", fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Swin Integrated Gradients
    attr_swin = compute_integrated_gradients(swin, input_tensor, target_cls, n_steps)
    attr_swin_np = attr_swin.squeeze(0).cpu().detach().numpy()
    attr_swin_np = np.transpose(attr_swin_np, (1, 2, 0))
    
    # Moyenne des canaux pour avoir une intensité globale
    heatmap_swin = np.mean(np.abs(attr_swin_np), axis=2)
    
    # Afficher l'original en fond pour Swin (focus attention)
    axes[2].imshow(original_img, alpha=0.6)
    im2 = axes[2].imshow(heatmap_swin, cmap='jet', alpha=0.5, vmin=0, vmax=np.max(heatmap_swin))
    axes[2].set_title(f"{title_prefix}Swin IG (Attention Focus)", fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_occlusion_comparison(convnext, swin, input_tensor, target_cls, original_img,
                              window_size=32, stride=16, title_prefix="", save_path=None):
    """
    Plot side-by-side Occlusion comparison
    
    Args:
        convnext: ConvNeXt model
        swin: Swin model
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
    axes[0].set_title(f"{title_prefix}Original\nTrue Label: {CLASS_NAMES[target_cls]}", fontsize=12)
    axes[0].axis('off')
    
    # ConvNeXt Occlusion
    attr_convnext = compute_occlusion(convnext, input_tensor, target_cls, window_size, stride)
    attr_convnext_np = attr_convnext.squeeze(0).cpu().detach().numpy()
    attr_convnext_np = np.transpose(attr_convnext_np, (1, 2, 0))
    
    # Transformation en Heatmap 2D
    heatmap_convnext = np.mean(np.abs(attr_convnext_np), axis=2)
    
    # Affichage
    axes[1].imshow(original_img, alpha=0.6)
    im1 = axes[1].imshow(heatmap_convnext, cmap='jet', alpha=0.5, vmin=0, vmax=np.max(heatmap_convnext))
    axes[1].set_title(f"{title_prefix}ConvNeXt Occlusion", fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Swin Occlusion
    attr_swin = compute_occlusion(swin, input_tensor, target_cls, window_size, stride)
    attr_swin_np = attr_swin.squeeze(0).cpu().detach().numpy()
    attr_swin_np = np.transpose(attr_swin_np, (1, 2, 0))
    
    # Transformation en Heatmap 2D
    heatmap_swin = np.mean(np.abs(attr_swin_np), axis=2)
    
    # Affichage
    axes[2].imshow(original_img, alpha=0.6)
    im2 = axes[2].imshow(heatmap_swin, cmap='jet', alpha=0.5, vmin=0, vmax=np.max(heatmap_swin))
    axes[2].set_title(f"{title_prefix}Swin Occlusion", fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# --- ERROR ANALYSIS ---

def find_prediction_cases(convnext, swin, dataloader, device='cuda', class_filter=None):
    """
    Find interesting prediction cases: CNN wrong/Swin right, vice versa, both wrong
    
    Args:
        convnext: ConvNeXt model
        swin: Swin model
        dataloader: Test dataloader
        device: Device
        class_filter: Optional class index to filter results (only analyze this class)
    
    Returns:
        Dictionary with lists of indices for each case:
        - 'cnn_wrong_swin_right': indices where CNN fails but Swin succeeds
        - 'swin_wrong_cnn_right': indices where Swin fails but CNN succeeds
        - 'both_wrong': indices where both models fail
        - 'both_right': indices where both models succeed
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    dataset = dataloader.dataset
    
    cases = {
        'cnn_wrong_swin_right': [],
        'swin_wrong_cnn_right': [],
        'both_wrong': [],
        'both_right': []
    }
    
    convnext.eval()
    swin.eval()
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            img_tensor, label = dataset[idx]
            
            # Apply class filter if specified
            if class_filter is not None and label != class_filter:
                continue
            
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            # Get predictions
            convnext_pred = convnext(img_tensor).argmax(dim=1).item()
            swin_pred = swin(img_tensor).argmax(dim=1).item()
            # label = label.item()
            

            # Categorize
            convnext_correct = (convnext_pred == label)
            swin_correct = (swin_pred == label)
            
            if convnext_correct and swin_correct:
                cases['both_right'].append(idx)
            elif not convnext_correct and swin_correct:
                cases['cnn_wrong_swin_right'].append(idx)
            elif convnext_correct and not swin_correct:
                cases['swin_wrong_cnn_right'].append(idx)
            else:
                cases['both_wrong'].append(idx)
    
    # Calculate total for percentage
    total = sum(len(v) for v in cases.values())
    
    # Print summary
    class_str = f" for class '{CLASS_NAMES[class_filter]}'" if class_filter is not None else ""
    print(f"\n{'='*60}")
    print(f"ImageNet-R-mini Analysis{class_str}:")
    print(f"{'='*60}")
    print(f"  Both correct: {len(cases['both_right'])} ({100*len(cases['both_right'])/total:.1f}%)")
    print(f"  CNN wrong, Swin right: {len(cases['cnn_wrong_swin_right'])} ({100*len(cases['cnn_wrong_swin_right'])/total:.1f}%)")
    print(f"  Swin wrong, CNN right: {len(cases['swin_wrong_cnn_right'])} ({100*len(cases['swin_wrong_cnn_right'])/total:.1f}%)")
    print(f"  Both wrong: {len(cases['both_wrong'])} ({100*len(cases['both_wrong'])/total:.1f}%)")
    print(f"{'='*60}\n")
    
    return cases

def analyze_sample(convnext, swin, dataloader, idx, device='cuda', methods=['gradcam', 'ig', 'occlusion']):
    """
    Complete analysis of a single sample with all interpretability methods
    
    Args:
        convnext: ConvNeXt model
        swin: Swin model
        dataloader: Test dataloader
        idx: Sample index
        device: Device
        methods: List of methods to apply ['gradcam', 'ig', 'occlusion']
    """
    # Load image
    input_tensor, label_idx, original_img = get_sample_from_dataloader(dataloader, idx, device)
    
    # Get predictions
    with torch.no_grad():
        convnext_pred = convnext(input_tensor).argmax(dim=1).item()
        swin_pred = swin(input_tensor).argmax(dim=1).item()
    
    title_prefix = f"Sample {idx} | True: {CLASS_NAMES[label_idx]} | ConvNeXt: {CLASS_NAMES[convnext_pred]} | Swin: {CLASS_NAMES[swin_pred]}\n"
    
    print(f"\n{'='*80}")
    print(f"Analyzing Sample {idx} from ImageNet-R-mini")
    print(f"Ground Truth: {CLASS_NAMES[label_idx]} ({label_idx})")
    print(f"ConvNeXt Prediction: {CLASS_NAMES[convnext_pred]} ({'✓' if convnext_pred == label_idx else '✗'})")
    print(f"Swin Prediction: {CLASS_NAMES[swin_pred]} ({'✓' if swin_pred == label_idx else '✗'})")
    print(f"{'='*80}\n")
    
    # Apply methods
    if 'gradcam' in methods:
        print("Computing GradCAM comparison...")
        plot_gradcam_comparison(convnext, swin, input_tensor, label_idx, original_img, title_prefix)
    
    if 'ig' in methods:
        print("Computing Integrated Gradients comparison...")
        plot_integrated_gradients_comparison(convnext, swin, input_tensor, label_idx, original_img, title_prefix=title_prefix)
    
    if 'occlusion' in methods:
        print("Computing Occlusion comparison...")
        plot_occlusion_comparison(convnext, swin, input_tensor, label_idx, original_img, title_prefix=title_prefix)
