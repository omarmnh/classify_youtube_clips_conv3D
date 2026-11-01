"""
extract_frames.py
Extraction des images et création du fichier manifeste pour UCF101.
Conforme aux exigences de la section 5.1(a) du Lab 5.
"""

import os
import csv
import glob
import numpy as np
import torch
from torchvision.io import read_video
import torchvision.transforms.functional as TF
from tqdm import tqdm


def resize_short_side(frames: torch.Tensor, target_short_side: int = 80) -> torch.Tensor:
    """
    Redimensionne un tenseur vidéo de forme (T, C, H, W) ou (T, H, W, C)
    pour que son côté court soit égal à target_short_side en conservant le ratio.
    """
    # torchvision read_video renvoie (T, H, W, C) avec des valeurs uint8 [0, 255]
    if frames.ndim == 4 and frames.shape[-1] == 3:
        # Passage à (T, C, H, W) pour le traitement torchvision
        frames = frames.permute(0, 3, 1, 2)
    
    _, _, h, w = frames.shape
    if h < w:
        new_h = target_short_side
        new_w = int(w * (target_short_side / h))
    else:
        new_w = target_short_side
        new_h = int(h * (target_short_side / w))
        
    # Utilisation de TF.resize (attend un tenseur (C, H, W) ou un lot)
    resized_frames = torch.stack([TF.resize(f, [new_h, new_w]) for f in frames])
    return resized_frames.permute(0, 2, 3, 1)  # Retour en (T, H, W, C) pour stockage uint8


def extract_and_save_video(video_path: str, output_npy_path: str, num_frames: int = 16, short_side: int = 80):
    """
    Décode la vidéo, échantillonne uniformément num_frames images, 
    redimensionne le côté court et sauvegarde en tant que fichier .npy uint8.
    """
    try:
        # read_video renvoie vframes: (T, H, W, C), aframes, info
        vframes, _, _ = read_video(video_path, pts_unit='sec')
        total_frames = vframes.shape[0]
        
        if total_frames == 0:
            print(f"Attention : Vidéo vide ou illisible -> {video_path}")
            return False

        # Échantillonnage uniforme de 16 images
        if total_frames >= num_frames:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            # Si la vidéo a moins de 16 images, on répète la dernière image ou on prend tout
            indices = np.pad(np.arange(total_frames), (0, num_frames - total_frames), mode='edge')
            
        sampled_frames = vframes[indices]
        
        # Redimensionnement (côté court à 80 pixels)
        resized_frames = resize_short_side(sampled_frames, target_short_side=short_side)
        
        # Conversion en numpy uint8
        np_frames = resized_frames.numpy().astype(np.uint8)
        
        # Sauvegarde au format .npy
        np.save(output_npy_path, np_frames)
        return True

    except Exception as e:
        print(f"Erreur lors du traitement de {video_path}: {e}")
        return False


def build_ucf101_manifest(data_root: str, output_dir: str, manifest_csv_path: str):
    """
    Parcourt le dossier UCF101, extrait les images de chaque vidéo, 
    et crée le fichier CSV manifeste contenant : video_path, class_label, group_id, split.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Structure type UCF101 : data_root/NomDeClasse/nom_de_video.avi
    class_folders = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    
    manifest_data = []
    
    for class_label in class_folders:
        class_path = os.path.join(data_root, class_label)
        video_files = glob.glob(os.path.join(class_path, "*.avi"))
        
        print(f"Traitement de la classe : {class_label} ({len(video_files)} vidéos)")
        
        for vid_path in tqdm(video_files):
            vid_name = os.path.basename(vid_path)
            vid_id_no_ext = os.path.splitext(vid_name)[0]
            
            # Extraction de l'ID du groupe à partir du nom UCF101 (ex: v_ApplyEyeMakeup_g01_c01.avi -> g01)
            group_id = "g01"
            if "_g" in vid_id_no_ext:
                parts = vid_id_no_ext.split("_")
                for p in parts:
                    if p.startswith("g") and p[1:].isdigit():
                        group_id = p
                        break
            
            # Sortie .npy correspondante
            npy_name = f"{class_label}_{vid_id_no_ext}.npy"
            npy_output_path = os.path.join(output_dir, npy_name)
            
            # Traitement et sauvegarde si non existant
            success = True
            if not os.path.exists(npy_output_path):
                success = extract_and_save_video(vid_path, npy_output_path)
            
            if success:
                # Attribution de la partition (train/val) basée sur le groupe pour éviter la fuite de données
                # Par exemple, les groupes g01 à g07 pour le train, g08+ pour la validation (selon le split officiel ou personnalisé)
                group_num = int(group_id[1:]) if group_id[1:].isdigit() else 1
                split = "val" if group_num >= 8 else "train"
                
                manifest_data.append({
                    "video_path": npy_output_path,
                    "class_label": class_label,
                    "group_id": group_id,
                    "split": split
                })

    # Écriture du fichier manifeste CSV
    with open(manifest_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "class_label", "group_id", "split"])
        writer.writeheader()
        writer.writerows(manifest_data)
        
    print(f"Manifeste généré avec succès : {manifest_csv_path} ({len(manifest_data)} entrées enregistrées).")


if __name__ == "__main__":
    # Paramètres par défaut à adapter selon l'emplacement de vos données
    UCF101_ROOT = "./UCF-101"  # Dossier contenant les dossiers de classes .avi
    PROCESSED_DIR = "./ucf101_processed_npys"  # Dossier de destination des fichiers .npy
    MANIFEST_CSV = "./ucf101_manifest.csv"
    
    # Lancement de la génération du manifeste et des fichiers npy
    # build_ucf101_manifest(UCF101_ROOT, PROCESSED_DIR, MANIFEST_CSV)
    print("Script `extract_frames.py` prêt. Configurez vos chemins et exécutez `build_ucf101_manifest(...)`.")