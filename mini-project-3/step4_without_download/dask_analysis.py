# %%
import dask.dataframe as dd
import pandas as pd
from PIL import Image
import io

splits = {'train': 'data/train-*.parquet', 'validation': 'data/validation-*.parquet', 'test': 'data/test-*.parquet'}
df_mini_imagenet = dd.read_parquet("hf://datasets/timm/mini-imagenet/" + splits["train"])

# %%
# check unique labels in the train split
unique_labels_mini_imagenet = df_mini_imagenet['label'].unique().compute()
print("Unique labels in train split:", unique_labels_mini_imagenet.tolist())

# %%
import dask.dataframe as dd

df_imagenet_r = dd.read_parquet("hf://datasets/axiong/imagenet-r/test/test-*.parquet")

df_imagenet_r.head(5)

# %%
# check unique labels in the train split
unique_labels_imagenet_r = df_imagenet_r['wnid'].unique().compute()
# print("Unique labels in train split:", unique_labels.tolist())
class_names_imagenet_r = df_imagenet_r['class_name'].unique().compute()

# %%
# Create a dictionary mapping wnid to class_name from ImageNet-R
wnid_to_class_name = dict(zip(df_imagenet_r['wnid'].compute(), df_imagenet_r['class_name'].compute()))
print(f"Created mapping for {len(wnid_to_class_name)} classes")

# %%
import pandas as pd

# 1. Dictionnaire Mini-ImageNet (Mapping label -> wnid)
mini_imagenet_dict = {
    '0': 'n01532829', '1': 'n01558993', '2': 'n01704323', '3': 'n01749939', '4': 'n01770081',
    '5': 'n01843383', '6': 'n01855672', '7': 'n01910747', '8': 'n01930112', '9': 'n01981276',
    '10': 'n02074367', '11': 'n02089867', '12': 'n02091244', '13': 'n02091831', '14': 'n02099601',
    '15': 'n02101006', '16': 'n02105505', '17': 'n02108089', '18': 'n02108551', '19': 'n02108915',
    '20': 'n02110063', '21': 'n02110341', '22': 'n02111277', '23': 'n02113712', '24': 'n02114548',
    '25': 'n02116738', '26': 'n02120079', '27': 'n02129165', '28': 'n02138441', '29': 'n02165456',
    '30': 'n02174001', '31': 'n02219486', '32': 'n02443484', '33': 'n02457408', '34': 'n02606052',
    '35': 'n02687172', '36': 'n02747177', '37': 'n02795169', '38': 'n02823428', '39': 'n02871525',
    '40': 'n02950826', '41': 'n02966193', '42': 'n02971356', '43': 'n02981792', '44': 'n03017168',
    '45': 'n03047690', '46': 'n03062245', '47': 'n03075370', '48': 'n03127925', '49': 'n03146219',
    '50': 'n03207743', '51': 'n03220513', '52': 'n03272010', '53': 'n03337140', '54': 'n03347037',
    '55': 'n03400231', '56': 'n03417042', '57': 'n03476684', '58': 'n03527444', '59': 'n03535780',
    '60': 'n03544143', '61': 'n03584254', '62': 'n03676483', '63': 'n03770439', '64': 'n03773504',
    '65': 'n03775546', '66': 'n03838899', '67': 'n03854065', '68': 'n03888605', '69': 'n03908618',
    '70': 'n03924679', '71': 'n03980874', '72': 'n03998194', '73': 'n04067472', '74': 'n04146614',
    '75': 'n04149813', '76': 'n04243546', '77': 'n04251144', '78': 'n04258138', '79': 'n04275548',
    '80': 'n04296562', '81': 'n04389033', '82': 'n04418357', '83': 'n04435653', '84': 'n04443257',
    '85': 'n04509417', '86': 'n04515003', '87': 'n04522168', '88': 'n04596742', '89': 'n04604644',
    '90': 'n04612504', '91': 'n06794110', '92': 'n07584110', '93': 'n07613480', '94': 'n07697537',
    '95': 'n07747607', '96': 'n09246464', '97': 'n09256479', '98': 'n13054560', '99': 'n13133613'
}

# 3. Trouver l'intersection (les classes présentes dans les deux)
# On convertit la liste imagenet-r en set pour une recherche rapide
imagenet_r_set = set(unique_labels_imagenet_r)

common_classes = []
for label, wnid in mini_imagenet_dict.items():
    if wnid in imagenet_r_set:
        common_classes.append({
            'class_name': wnid_to_class_name[wnid],
            'label': int(label),
            'wnid': wnid
        })

# 4. Créer le DataFrame avec les 20 premières classes trouvées
df_crossed = pd.DataFrame(common_classes)
# df_crossed.to_csv("step4/mini-imagenet_imagenet-r_crossed_classes.csv", index=False)

# Affichage du résultat
print(df_crossed)

# %%
row1 = df_mini_imagenet[df_mini_imagenet['label'] == 5].compute().head(1)

# %%
row2 =df_imagenet_r[df_imagenet_r['wnid'] == 'n01843383'].compute().head(1)

# %%
import matplotlib.pyplot as plt
import io
from PIL import Image
import ast

# Fonction simple pour afficher une paire d'images
def afficher_paire(wnid, label):
    """
    Affiche côte à côte une image de mini-imagenet et imagenet-r
    wnid: le WNID (ex: 'n01843383')
    label: le label mini-imagenet (ex: 5 ou '5')
    """
    # Récupérer le class_name depuis df_crossed
    row_info = df_crossed[df_crossed['wnid'] == wnid]
    if len(row_info) > 0:
        class_name = row_info.iloc[0]['class_name']
    else:
        class_name = wnid
    
    
    # Chercher dans mini-imagenet
    mini_sample = df_mini_imagenet[df_mini_imagenet['label'] == label].compute().head(1)
    if len(mini_sample) > 0:
        mini_img_bytes = mini_sample.iloc[0]['image']
        mini_img_bytes = ast.literal_eval(mini_img_bytes)['bytes']
        mini_image = Image.open(io.BytesIO(mini_img_bytes)).convert("RGB")
    else:
        mini_image = None
        print(f"⚠️  Aucune image trouvée dans mini-imagenet pour label={label}")
    
    # Chercher dans imagenet-r
    imagenet_r_sample = df_imagenet_r[df_imagenet_r['wnid'] == wnid].compute().head(1)
    if len(imagenet_r_sample) > 0:
        imagenet_r_img_bytes = imagenet_r_sample.iloc[0]['image']
        imagenet_r_img_bytes = ast.literal_eval(imagenet_r_img_bytes)['bytes']
        imagenet_r_image = Image.open(io.BytesIO(imagenet_r_img_bytes)).convert("RGB")
    else:
        imagenet_r_image = None
        print(f"⚠️  Aucune image trouvée dans imagenet-r pour wnid={wnid}")
    
    # Afficher
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    if mini_image:
        axes[0].imshow(mini_image)
        axes[0].set_title(f"Mini-ImageNet\nLabel: {label}")
        axes[0].axis('off')
    else:
        axes[0].text(0.5, 0.5, 'No image', ha='center', va='center')
        axes[0].axis('off')
    
    if imagenet_r_image:
        axes[1].imshow(imagenet_r_image)
        axes[1].set_title(f"ImageNet-R\nWNID: {wnid}")
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'No image', ha='center', va='center')
        axes[1].axis('off')
    
    fig.suptitle(f"Class: {class_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Tester avec le premier exemple
afficher_paire('n01843383', 5)

# %%
try:
    from dataloader import get_dataloaders
except ImportError:
    from step4_without_download.dataloader import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(batch_size=4, num_workers=0)

# %%
import matplotlib.pyplot as plt
import torchvision

for images, labels in train_loader:
    # images: [B, 3, 224, 224]
    # labels: [B] -> valeurs de 0 à 19
    print(images.size())
    print(labels.size())
    # Afficher les images
    grid_img = torchvision.utils.make_grid(images, nrow=4)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
    break

# %%



