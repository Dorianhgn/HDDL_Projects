import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=0, act_norm=False):
        super(BasicConv2d,self).__init__()
        self.act_norm = act_norm

        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size, stride,padding=padding)
        
        self.norm = nn.GroupNorm(2,out_channels)
        self.act = nn.SiLU(inplace=True) # SiLU = Swish for smooth activation function

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.norm(y)
            y = self.act(y)
        return y

class ConvSC(nn.Module):
    """
    Spacial-Convolution block that forms the backbone of the Encoder and Decoder.
    This is essentially a wrapper around BasicConv2d to handle stride and padding automatically.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, downsampling=False, upsampling=False, act_norm=True):
        super().__init__()
        stride = 2 if downsampling else 1
        padding = (kernel_size - stride + 1) // 2
        self.conv = BasicConv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, act_norm=act_norm
        )

    def forward(self, x):
        return self.conv(x)
class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1):
        super(DoubleConv,self).__init__()
        self.double_conv = nn.Sequential(
            BasicConv2d(in_channels,out_channels,kernel_size,stride=stride,padding=1, act_norm=True), #stride =2 dans encoder pour réduire la taille
            BasicConv2d(out_channels,out_channels,kernel_size,stride=1,padding=1, act_norm=True) #stride=1 toujours pour affiner les features
        )

    def forward(self, x):
        return self.double_conv(x)

class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, n_s):
        """
        Args:
            C_in: Canaux image entrée (3)
            C_hid: Canaux cachés initiaux (ex: 64)
            n_s: Nombre de descentes (ex: 4)
        """
        super().__init__()
        
        # --- Instruction Init ---
        # self.enc = nn.Sequential(DoubleConv(stride=1), *[DoubleConv(stride=2) for _ in range(n_s-1)])
        
        layers = []
        # 1. Première couche (Garde la taille, stride=1)
        layers.append(DoubleConv(C_in, C_hid, stride=1))
        
        # 2. Les couches suivantes (Réduisent la taille, stride=2)
        # On double les canaux à chaque descente : 64 -> 128 -> 256...
        c_in = C_hid
        for _ in range(n_s - 1):
            layers.append(DoubleConv(c_in, c_in * 2, stride=2))
            c_in *= 2
            
        # L'étoile * permet de "déballer" la liste dans le Sequential
        self.enc = nn.Sequential(*layers)
        self.final_channels = c_in # On garde en mémoire le nombre de canaux en bas

    def forward(self, x):
        skips = [] # Liste pour stocker les skip connections
        
        # "pour chaque layer de self.enc, on calcule x = layer(x) et on skip.append(x)"
        for layer in self.enc:
            x = layer(x)
            skips.append(x)
            
        # "on renvoie le x final et les skips connection"
        return x, skips
    
class Decoder(nn.Module):
    def __init__(self, C_hid_enc, n_classes, n_s):
        super().__init__()
        
        # INTERPRÉTATION : Dans un décodeur, "stride=2" signifie qu'on AGRANDIT.
        # On utilise donc ConvTranspose2d pour l'agrandissement + DoubleConv2d pour le traitement.
        
        self.dec = nn.ModuleList()  # liste pour stocker les couches du décodeur
        
        # On part du bas (nombre de canaux max de l'encodeur)
        c_curr = C_hid_enc 
        
        for _ in range(n_s - 1): 
            # o, agrandissement par 2 (stride=2), pour ceci on utilise conv transpose
            # DOuble conv avec stride=1 sert a mélanger les canaux après concaténation avec le skip connection
            self.dec.append(
                nn.ModuleDict({
                    'up': nn.ConvTranspose2d(c_curr, c_curr // 2, kernel_size=2, stride=2),
                    'conv': DoubleConv(c_curr, c_curr // 2, stride=1) 
                })
            )
            c_curr //= 2
            
        # "nn.final_conv : 1 final Double_Conv avec stride = 1"
        self.final_conv = DoubleConv(c_curr, c_curr, stride=1)
        
        # "nn.readout = nn.Conv2d(C_hid, C_out, 1)"
        self.readout = nn.Conv2d(c_curr, n_classes, kernel_size=1)

    def forward(self, hid, skips):

        skips = skips[:-1][::-1] # On enlève le dernier skip car c'est l'image de départ du décodeur (bas de U) et on inverse l'ordre

        for i, layer in enumerate(self.dec):
  
            hid = layer['up'](hid)  # l'image hid est agrandie
            
            skip = skips[i] # on cherche le skip connection (souvenir) correspondant
            
            # (Sécurité optionnelle pour le padding si les tailles ne matchent pas parfaitement)
            if hid.shape != skip.shape:
                hid = F.interpolate(hid, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            # Concaténation (Standard U-Net)
            # En Pytorch "ajouter" une connexion skip se fait souvent par concaténation (dim=1)
            hid = torch.cat([skip, hid], dim=1) # on concatène le skip connection avec hid (on somme les canaux)
            
            # Double Conv
            hid = layer['conv'](hid) # on remet le bon nombre de canaux avec une double conv
            
       # A cette etape on a fini les étapes d'up-sampling
        hid = self.final_conv(hid) # final double conv pour lisser les canaux tels quels
        y = self.readout(hid)   # pour passer de 64 canaux à n_classes canaux
        
        return y
class UNet(nn.Module):
    def __init__(self, n_classes, n_s=4, C_in=3, C_hid=64):
        super().__init__()
        self.encoder = Encoder(C_in, C_hid, n_s)
        
        # On récupère la taille de sortie de l'encodeur pour configurer le décodeur
        C_bottom = self.encoder.final_channels
        
        self.decoder = Decoder(C_bottom, n_classes, n_s)

    def forward(self, x):
        # 1. Encoder
        x_final, skips = self.encoder(x)
        
        # 2. Decoder
        output = self.decoder(x_final, skips)
        
        return output
    
class attention_gate(nn.Module): 
    """
    cette classe implemente un gate d'attention pour le unet attention
    """
    def __init__(self, F_g, F_l, F_int):
        super(attention_gate,self).__init__()
        """
        Args: F_g : nombre de canaux du tenseur de gating (venant du décodeur)
              F_l : nombre de canaux du tenseur de skip connection (venant de l'encodeur)
              F_int : nombre de canaux intermédiaires pour le calcul de l'attention

        """
        # W_g et W_x sont des convolutions 1x1 pour réduire le nombre de canaux 
        # psi est une convolution 1x1 suivie d'une sigmoid pour calculer la carte d'attention
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(2, F_int) # c le batchnorm pour les petits batchs
        )
        

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(2, F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),  # 1 seul canal, on utilise BatchNorm au lieu de GroupNorm
            nn.Sigmoid()  # activation pour obtenir des valeurs entre 0 et 1
        )

        self.relu = nn.SiLU(inplace=True) # SILU active les zones importantes

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # appliquer la carte d'attention au skip connection 
    

class AttentionDecoder(nn.Module):
    def __init__(self, C_hid_enc, n_classes, n_s):
        super().__init__()
        
        self.dec = nn.ModuleList()  # liste pour stocker les couches du décodeur
        
        c_curr = C_hid_enc   
        
        for _ in range(n_s - 1): 
            filters = c_curr // 2 # nombre de filres voulus  
            self.dec.append(
                nn.ModuleDict({
                    'up': nn.ConvTranspose2d(c_curr, filters, kernel_size=2, stride=2),
                    'attn': attention_gate(F_g=filters, F_l=filters, F_int=filters // 2),
                    'conv': DoubleConv(c_curr, filters, stride=1) 
                })
            )
            c_curr //= 2
            
        self.final_conv = DoubleConv(c_curr, c_curr, stride=1)
        self.readout = nn.Conv2d(c_curr, n_classes, kernel_size=1)

    def forward(self, hid, skips):

        skips = skips[:-1][::-1]   #on inverse pour remonter 

        for i, layer in enumerate(self.dec):
  
            g = layer['up'](hid)  
            
            x = skips[i] 
            
            if g.shape != x.shape:
                g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
            
            x_att = layer['attn'](g, x) # x_att ne contient que les parties importantes du skip connection 
            
            hid = torch.cat([x_att, g], dim=1) 
            
            hid = layer['conv'](hid) 
            
       # A cette etape on a fini les étapes d'up-sampling
        hid = self.final_conv(hid) 
        y = self.readout(hid)   
        
        return y

class AttentionUNet(nn.Module):
    def __init__(self, n_classes, n_s=4, C_in=3, C_hid=64):
        super().__init__()
        self.encoder = Encoder(C_in, C_hid, n_s)
        
        C_bottom = self.encoder.final_channels
        
        self.decoder = AttentionDecoder(C_bottom, n_classes, n_s)

    def forward(self, x):
        x_final, skips = self.encoder(x)
        
        output = self.decoder(x_final, skips)
        
        return output