import torch
import argparse
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import CLEVR_ReasoningDataset
# On importe le model factory et la fonction evaluate du fichier train
from train import get_model, evaluate

def test_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # 1. Prepare Data
    # Pour le test, on utilise souvent le set de validation ou un set de test dédié
    # Ici j'utilise le json de validation de la tâche demandée
    json_val = f"task{args.task}_val.json"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Attention: img_dir doit pointer vers le dossier qui contient les images de validation si elles sont séparées
    # Dans CLEVR standard, train et val sont séparés.
    test_set = CLEVR_ReasoningDataset(json_val, args.img_dir, transform=transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. Load Model
    print(f"Loading checkpoint: {args.ckpt_path}")
    # On force pretrained=False car on va charger nos propres poids
    model = get_model(args.model_type, num_classes=8, pretrained=False)
    
    # Chargement des poids
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    # 3. Evaluate
    print("Starting evaluation...")
    avg_loss, accuracy = evaluate(model, test_loader, device)
    
    print(f"Result -> Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # 4. Save Results to .txt
    # On récupère le dossier parent du checkpoint
    exp_dir = os.path.dirname(args.ckpt_path)
    # On récupère le nom du fichier .pth pour nommer le .txt pareil
    ckpt_name = os.path.basename(args.ckpt_path)
    txt_name = ckpt_name.replace(".pth", "_results.txt")
    save_path = os.path.join(exp_dir, txt_name)
    
    with open(save_path, "w") as f:
        f.write(f"Checkpoint: {args.ckpt_path}\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Model: {args.model_type}\n")
        f.write("-" * 20 + "\n")
        f.write(f"Validation Loss: {avg_loss:.4f}\n")
        f.write(f"Validation Accuracy: {accuracy:.2f}%\n")
    
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the .pth file")
    parser.add_argument("--model_type", type=str, required=True, choices=['resnet50', 'vit_b_16'])
    parser.add_argument("--task", type=int, required=True, choices=[1, 2])
    # Attention au chemin par défaut des images val
    parser.add_argument("--img_dir", type=str, default="data_clevr/CLEVR_v1.0/images/val")
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    test_pipeline(args)