"""
data_loader.py
Pipeline de données pour UCF101 (Lab 5).
Gère le chargement des clips .npy, les augmentations (crop aléatoire, flip, jitter temporel)
pour l'entraînement, et les transformations de validation/test.
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


class UCF101Dataset(Dataset):
    """
    Dataset PyTorch pour le chargement des clips vidéo UCF101 prétraités en .npy.
    """
    def __init__(self, manifest_csv: str, split: str = "train", transform=None):
        """
        Args:
            manifest_csv (str): Chemin vers le fichier CSV manifeste.
            split (str): "train", "val" ou "test".
            transform (callable, optional): Transformations personnalisées optionnelles.
        """
        self.samples = []
        self.split = split
        self.transform = transform
        
        # Mappage des classes vers des entiers (0 à 101)
        self.classes = []
        self._load_manifest(manifest_csv)

    def _load_manifest(self, manifest_csv: str):
        """Charge le manifeste et filtre par split (train/val/test)."""
        classes_set = set()
        temp_samples = []

        with open(manifest_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == self.split:
                    temp_samples.append({
                        "video_path": row["video_path"],
                        "class_label": row["class_label"],
                        "group_id": row["group_id"]
                    })
                    classes_set.add(row["class_label"])

        self.classes = sorted(list(classes_set))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for sample in temp_samples:
            self.samples.append({
                "video_path": sample["video_path"],
                "label": self.class_to_idx[sample["class_label"]]
            })
            
        print(f"Chargement de la partition [{self.split}] : {len(self.samples)} clips trouvés, {len(self.classes)} classes.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        video_path = sample["video_path"]
        label = sample["label"]

        # Chargement du tableau numpy uint8 de forme (T, H, W, C) [ex: (16, 80, 80, 3)]
        try:
            frames = np.load(video_path)
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du fichier .npy {video_path}: {e}")

        # Conversion en tenseur PyTorch et permutation en (T, C, H, W)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        # Application des augmentations selon le split
        if self.split == "train":
            frames = self._train_augmentations(frames)
        else:
            frames = self._eval_augmentations(frames)

        # Le modèle attend un tenseur de forme (C, T, H, W)
        frames = frames.permute(1, 0, 2, 3)

        return frames, torch.tensor(label, dtype=torch.long)

    def _train_augmentations(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applique les augmentations d'entraînement :
        - Jitter temporel aléatoire (décalage de ±2 images)
        - Crop aléatoire à 64x64 depuis le 80x80
        - Flip horizontal aléatoire
        """
        T, C, H, W = frames.shape
        crop_size = 64

        # 1. Jitter temporel aléatoire (±2 images si possible)
        if T > 16: # Sécurité si plus de 16 images, ou décalage dans la séquence
            pass
        # Simulation d'un décalage temporel aléatoire (shift de -2 à +2 images)
        shift = np.random.randint(-2, 3)
        if shift != 0:
            frames = torch.roll(frames, shifts=shift, dims=0)

        # 2. Crop aléatoire 64x64 (à partir de 80x80)
        if H > crop_size or W > crop_size:
            top = np.random.randint(0, H - crop_size + 1)
            left = np.random.randint(0, W - crop_size + 1)
            frames = frames[:, :, top:top+crop_size, left:left+crop_size]
        else:
            # Fallback si dimensions inférieures
            frames = TF.resize(frames.permute(1, 2, 3, 0), [crop_size, crop_size]).permute(3, 0, 1, 2)

        # 3. Flip horizontal aléatoire (appliqué uniformément sur toutes les images du clip)
        if np.random.rand() > 0.5:
            frames = torch.flip(frames, dims=[3])  # Flip sur l'axe de la largeur (W)

        return frames

    def _eval_augmentations(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applique les transformations pour la validation/test :
        - Crop centré à 64x64 (pas d'augmentation)[cite: 3]
        """
        T, C, H, W = frames.shape
        crop_size = 64

        # Crop centré 64x64
        if H >= crop_size and W >= crop_size:
            top = (H - crop_size) // 2
            left = (W - crop_size) // 2
            frames = frames[:, :, top:top+crop_size, left:left+crop_size]
        else:
            frames = TF.resize(frames.permute(1, 2, 3, 0), [crop_size, crop_size]).permute(3, 0, 1, 2)

        return frames


def get_data_loaders(manifest_csv: str, batch_size: int = 32, num_workers: int = 4):
    """
    Crée et retourne les DataLoaders PyTorch pour l'entraînement et la validation.
    """
    train_dataset = UCF101Dataset(manifest_csv=manifest_csv, split="train")
    val_dataset = UCF101Dataset(manifest_csv=manifest_csv, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader, len(train_dataset.classes)


if __name__ == "__main__":
    # Test rapide du Data Loader
    MANIFEST_CSV = "./ucf101_manifest.csv"
    if os.path.exists(MANIFEST_CSV):
        train_loader, val_loader, num_classes = get_data_loaders(MANIFEST_CSV, batch_size=4, num_workers=0)
        print(f"Nombre de classes détectées : {num_classes}")
        
        for clips, labels in train_loader:
            print(f"Forme d'un lot de clips (entraîner) : {clips.shape}")  # Attendu : torch.Size([4, 3, 16, 64, 64])
            print(f"Forme des étiquettes : {labels.shape}")          # Attendu : torch.Size([4])
            break
    else:
        print(f"Fichier manifeste non trouvé à l'emplacement : {MANIFEST_CSV}. Exécutez d'abord extract_frames.py.")