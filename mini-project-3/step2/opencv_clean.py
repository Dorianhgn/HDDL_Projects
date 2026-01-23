import os
import shutil
import glob

# 1. Créer un dossier de "quarantaine"
backup_dir = "/usr/local/lib/opencv_backup_sys"
if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)
    print(f"Dossier de backup créé : {backup_dir}")

# 2. Identifier les fichiers conflictuels (seulement les libopencv*)
# Attention : on ne touche pas à libtorchvision ou aux autres !
libs_to_move = glob.glob("/usr/local/lib/libopencv*")

print(f"Nombre de fichiers système OpenCV trouvés : {len(libs_to_move)}")

# 3. Les déplacer
count = 0
for lib in libs_to_move:
    try:
        file_name = os.path.basename(lib)
        dst = os.path.join(backup_dir, file_name)
        shutil.move(lib, dst)
        count += 1
    except Exception as e:
        print(f"Erreur sur {lib}: {e}")

print(f"Terminé : {count} fichiers déplacés dans {backup_dir}")
print("Les bibliothèques système sont maintenant cachées.")