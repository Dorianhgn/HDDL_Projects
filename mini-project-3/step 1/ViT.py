""" Image Patching and Embedding
    Positional Encoding
    Transformer Encoder
    Classification Head (MLP Head)"""

import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image Patching and Embedding ------------------------------------------------------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, hidden_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: (batch_size, in_channels, img_size, img_size)
        x = self.proj(x)  # Shape: (batch_size, hidden_dim, num_patches_sqrt, num_patches_sqrt)
        x = x.flatten(2)  # Shape: (batch_size, hidden_dim, num_patches)
        x = x.transpose(1, 2)  # Shape: (batch_size, num_patches, hidden_dim)
        return x
    
# Positional encoding  ----------------------------------------------------------------------------------------------------------------------------

# class PositionalEncoding(nn.Module): # à enlever on utilise un paramètre learnable directement dans le VIT
#     def __init__(self, num_patches, hidden_dim):
#         super().__init__()
#         self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, hidden_dim)) # uniquement positional embeding pour VIT (learnable)

#     def forward(self, x):
#         x = x + self.pos_embedding # patch embbeded + positional info 
#         return x

# Transformer Encoder ---------------------------------------------------------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        #weight matrices
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.query(query)  # (batch_size, seq_len, hidden_dim)
        K = self.key(key)      # (batch_size, seq_len, hidden_dim)
        V = self.value(value)  # (batch_size, seq_len, hidden_dim)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, seq_len, seq_len)

        if mask is not None: 
            scores = scores.masked_fill(mask == 0, float('-inf'))   # pas besoin de mask ici pour le VIT

        attn_weights = torch.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)  # (batch_size, seq_len, hidden_dim)

        output = self.out(attn_output)  # (batch_size, seq_len, hidden_dim)
        return output
    
# class LayerNorm(nn.Module): # à enlever on utilise nn.LayerNorm de pytorch
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.eps = 1e-5 # small value to avoid division by zero
#         self.scale = nn.Parameter(torch.ones(hidden_dim)) # scale parameter (learnable)
#         self.shift = nn.Parameter(torch.zeros(hidden_dim)) # shift parameter (learnable)

#     def forward(self, x):
#         mean = x.mean(dim=-1, keepdim=True)
#         var = x.var(dim=-1, keepdim=True, unbiased=False)
#         norm_x = (x - mean) / torch.sqrt(var + self.eps)
#         y = self.scale * norm_x + self.shift
#         return y
    
class FeedForward(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),  #regarder pourquoi mieux que ReLU ici 
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x):
        return self.layers(x)
    
class TransformerEncoderBlock(nn.Module):
  def __init__(self, hidden_dim, num_heads):
    super().__init__()

    self.attention = MultiHeadAttention(hidden_dim, num_heads)
    self.norm1 = nn.LayerNorm(hidden_dim)
    self.norm2 = nn.LayerNorm(hidden_dim)
    self.feed_forward = FeedForward(hidden_dim)

  def forward(self, x, mask=None): 
    #x=self.norm1(x)
    #x = x + self.attention(x,x,x) # skip connection et attention
    attn_out = self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
    x = x + attn_out  # skip connection
    x=self.norm2(x)
    x = x + self.feed_forward(x) # skip connection et feed forward
    return x
  
# Classification Head (MLP Head) ------------------------------------------------------------------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, hidden_dim)
        x = self.mlp(x)  # Shape: (batch_size, num_classes) 
        return x
    
# Créartion du VIT ------------------------------------------------------------------------------------------------------------------------------------------------


class VisionTransformer(nn.Module):
    def __init__(self, 
                 img_size=28, # à adapter au dataset
                 patch_size=4, 
                 in_channels=1, 
                 hidden_dim=128, 
                 num_heads=8, # nbr de têtes d'attention (par layer)
                 num_layers=6, # nbr de blocks transformer empilés 
                 num_classes=10): #à modifier selon le dataset choisi 
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))  # Token de classification (learnable)
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.patch_embedding.num_patches + 1, hidden_dim)) # +1 pour le CLS token
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, num_heads) for _ in range(num_layers) #
        ])
        self.dropout = nn.Dropout(0.1) # taux de dropout pour éviter l'overfitting
        self.classification_head = ClassificationHead(hidden_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x) # embedding des patches

        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, hidden_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: (batch_size, num_patches + 1, hidden_dim)
        
        x = x + self.positional_encoding # ajout du positional encoding (batch_size, num_patches + 1, hidden_dim)

        x=self.dropout(x)  #empeche overfitting 

        for block in self.transformer_blocks:
            x = block(x)

        x = x[:, 0]  # Shape: (batch_size, hidden_dim) Extraction du cls token 

        x = self.classification_head(x)  # Shape: (batch_size, num_classes) Classification 
        return x