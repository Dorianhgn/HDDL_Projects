# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3.12 (torch)
#     language: python
#     name: hddltorch
# ---

# %% [markdown]
# ## Définition : Conditional Variational Autoencoder (CVAE)
#
# Le **Conditional Variational Autoencoder (CVAE)** est une extension du VAE qui permet la **génération de données conditionnelle**.
#
# En plus d'encoder la donnée d'entrée $x$ dans un espace latent $z$, le CVAE intègre une information supplémentaire, appelée **condition $c$**, à la fois dans les réseaux de l'**encodeur** et du **décodeur**.
#
# * **Objectif** : Apprendre la distribution conditionnelle $P(x | c)$ plutôt que la distribution non conditionnelle $P(x)$.
# * **Encodage** : Le modèle apprend un espace latent $z$ conditionné par $c$. L'encodeur modélise la distribution variationnelle $Q(z | x, c)$.
# * **Décodage** : Le décodeur utilise l'échantillon latent $z$ **et** la condition $c$ pour reconstruire la donnée d'entrée ou générer une nouvelle donnée $\hat{x}$. Le décodeur modélise la distribution $P(x | z, c)$.
#
#
#
# ---
#
# ## Différence clé avec le VAE
#
# La différence essentielle réside dans la **capacité à contrôler la génération** :
#
# | Caractéristique | Variational Autoencoder (VAE) | Conditional Variational Autoencoder (CVAE) |
# | :--- | :--- | :--- |
# | **Objectif** | Apprendre la distribution non conditionnelle $P(x)$. | Apprendre la **distribution conditionnelle** $P(x \| c)$. |
# | **Génération** | Génère une donnée $\hat{x}$ basée **uniquement** sur un point $z$ de l'espace latent (génération non contrôlée). | Génère une donnée $\hat{x}$ basée sur un point $z$ **ET** une **condition $c$ spécifiée** (génération contrôlée). |
# | **Décodeur** | Reçoit seulement l'échantillon latent $z$. | Reçoit l'échantillon latent $z$ **ET la condition $c$**. |
# | **Formulation** | Encodeur : $Q(z \| x)$ ; Décodeur : $P(x \| z)$ | Encodeur : $Q(z \| x, c)$ ; Décodeur : $P(x \| z, c)$ |
#
# En résumé, le **CVAE** permet de spécifier les **attributs** de la donnée générée (par exemple, générer un chiffre **3** ou une image de chien **assis**), tandis que le **VAE** génère des échantillons sans contrôle direct sur leurs caractéristiques.
#
#

# %%

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
from torchvision import datasets, transforms
batch_size = 128

# Data loading
transform = transforms.Compose([transforms.ToTensor()])
train_dataset_x = datasets.FashionMNIST(root='../data', train=True, transform=transform, download=True)
test_dataset_x = datasets.FashionMNIST(root='../data', train=False, transform=transform, download=True)

train_labels = train_dataset_x.targets
test_labels = test_dataset_x.targets

# One-hot encoding of labels
num_classes = 10
train_labels_onehot = F.one_hot(train_labels, num_classes=num_classes).float()
test_labels_onehot = F.one_hot(test_labels, num_classes=num_classes).float()

train_dataset = torch.utils.data.TensorDataset(train_dataset_x.data.float()/255.0, train_labels_onehot)
test_dataset = torch.utils.data.TensorDataset(test_dataset_x.data.float()/255.0, test_labels_onehot)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# %%
# print an example of train_dataset_x shape

print(train_dataset_x.data[0].shape)  # should be [28, 28]
print(train_labels_onehot[0])  # should be a one-hot vector of size 10

# %%
# check the images shape

examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_data.shape)  # should be [batch_size, 1, 28, 28]
print(example_targets.shape)  # should be [batch_size]


# %%
class CVAE(nn.Module):
    def __init__(self,latent_dim=10):
        super(CVAE,self).__init__()
        self.latent_dim = latent_dim 

        #Encoder 
        self.encoder = nn.Sequential (
            nn.Conv2d(1,32,kernel_size =4,stride=2,padding=1) #32 filtres de taille 4*4, stride 2, padding 1
            ,nn.BatchNorm2d(32)
            ,nn.ReLU()
            ,nn.Conv2d(32,64,kernel_size =4,stride=2,padding=1) #64 filtres de taille 4*4, stride 2,
            ,nn.BatchNorm2d(64)
            ,nn.ReLU()
            ,nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1)
            ,nn.BatchNorm2d(128)
            ,nn.ReLU()
        )

        self.fc_mu = nn.Linear(128 * 4 * 4+10, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4+10, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4 - 10)


        #Decoder 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (64, 8, 8)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 16, 16)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # (1, 32, 32)
            nn.Sigmoid(),
        )

    def encode(self, x, c):
        x = self.encoder(x) # batch_size=128, 128, 4, 4
        x = x.view(x.size(0), -1) # batch_size, 128*4*4
        x = torch.cat((x,c),1) # batch_size, 128*4*4 + 10
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode(self, z, c):  # z de taille 10
        x = self.fc_decode(z)  #  x de taille batch_size, 128*4*4 - 10
        x = torch.cat((x,c), 1) # x de taille batch_size, 128*4*4
        x = x.view(x.size(0), 128, 4, 4) # de taille batch_size,128,4,4
        x = self.decoder(x) # x de taille 128,4,4 à l'entrée du décodeur
        # Cropping pour obtenir exactement (1, 28, 28)
        x = x[:, :, 2:30, 2:30]
        return x

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.sample(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar  # Retourner mu et logvar pour la loss


# %%
LOSS = 'BCE'  # 'BCE' or 'MSE'

def loss_function(recon_x, x, mu, logvar, beta=1):
    if LOSS == 'BCE':
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    elif LOSS == 'MSE':
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    else:
        raise ValueError("Invalid LOSS type. Choose 'BCE' or 'MSE'.")
    
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * KLD


# %%
# Hyperparameters
latent_dim = 2
learning_rate = 1e-3
epochs = 10
beta = 1

# TODO: Initialize the VAE model and the Adam optimizer
# and move the model to the device

model = CVAE(latent_dim=latent_dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# TODO: Train the model for the given number of epochs

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, c) in enumerate(train_loader):
        data = data.to(device)
        c = c.to(device)
        # print(f"{data.shape}, {c.shape}")
        data = data.unsqueeze(1)  # Ajouter une dimension de canal
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, c)
        loss = loss_function(recon_batch, data, mu, logvar, beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')


# %%
def image_comparison(original, reconstructed, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original images
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].squeeze(), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.show()


# %%
random_images = next(iter(test_loader))[0][:10].to(device)
random_images = random_images.unsqueeze(1)  # Ajouter une dimension de canal
with torch.no_grad():
    reconstructed_images, _, _ = model(random_images, test_labels_onehot[:10].to(device))


image_comparison(random_images.cpu(), reconstructed_images.cpu(), n=10)

# %% [markdown]
# ## Image generation with CVAE

# %%
