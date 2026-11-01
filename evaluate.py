"""
evaluate.py
Script d'évaluation pour le modèle 3D CNN sur UCF101 (Lab 5).
Calcule l'exactitude au niveau du clip et de la vidéo, la matrice de confusion,
les courbes d'apprentissage et la comparaison des modèles.
"""

import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from model3d import VideoCNN3D
from data_loader import UCF101Dataset


def load_video_clips_for_evaluation(video_path: str, num_clips: int = 5, crop_size: int = 64):
    """
    Charge un fichier .npy de vidéo et extrait plusieurs clips (ex: 5 clips espacés uniformément)
    pour l'évaluation au niveau de la vidéo, ou un clip central pour l'évaluation au niveau du clip.
    """
    try:
        frames = np.load(video_path)  # Forme : (T, H, W, C)
    except Exception as e:
        print(f"Erreur de chargement pour {video_path}: {e}")
        return None

    # Conversion en tenseur PyTorch et permutation en (C, T, H, W)
    frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
    C, T, H, W = frames.shape

    clips = []
    
    if num_clips == 1:
        # Un seul clip centré temporellement et spatialement
        start_t = max(0, (T - 16) // 2)
        clip = frames[:, start_t:start_t+16, :, :]
        # Si la vidéo a moins de 16 images, on complète par padding
        if clip.shape[1] < 16:
            pad_len = 16 - clip.shape[1]
            clip = torch.cat([clip, clip[:, -1:].repeat(1, pad_len, 1, 1)], dim=1)
        
        # Center crop spatial (64x64)
        top = (H - crop_size) // 2
        left = (W - crop_size) // 2
        clip = clip[:, :, top:top+crop_size, left:left+crop_size]
        clips.append(clip)
    else:
        # 5 clips espacés uniformément le long de la vidéo
        start_indices = np.linspace(0, max(0, T - 16), num_clips, dtype=int)
        top = (H - crop_size) // 2
        left = (W - crop_size) // 2

        for start_t in start_indices:
            clip = frames[:, start_t:start_t+16, :, :]
            if clip.shape[1] < 16:
                pad_len = 16 - clip.shape[1]
                clip = torch.cat([clip, clip[:, -1:].repeat(1, pad_len, 1, 1)], dim=1)
            
            clip = clip[:, :, top:top+crop_size, left:left+crop_size]
            clips.append(clip)

    return torch.stack(clips)  # Forme : (num_clips, C, T, H, W)


def evaluate_model(manifest_csv: str, checkpoint_path: str, temporal: bool = True):
    """
    Évalue le modèle sur le jeu de test au niveau du clip et de la vidéo.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Évaluation du modèle ({'3D' if temporal else '2D'}) sur le périphérique : {device}")

    # Chargement du dataset de test
    test_dataset = UCF101Dataset(manifest_csv=manifest_csv, split="test")
    num_classes = len(test_dataset.classes)

    # Chargement du modèle entraîné
    model = VideoCNN3D(num_classes=num_classes, dropout=0.5, temporal=temporal).to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Point de contrôle chargé avec succès : {checkpoint_path} (Précision Val : {checkpoint.get('val_acc', 0)*100:.2f}%)")
    else:
        print(f"Attention : Aucun point de contrôle trouvé à {checkpoint_path}. Utilisation des poids aléatoires.")

    model.eval()

    y_true = []
    clip_preds = []
    video_preds = []

    print("Début de l'évaluation sur le jeu de test...")
    with torch.no_grad():
        for i in range(len(test_dataset.samples)):
            sample = test_dataset.samples[i]
            video_path = sample["video_path"]
            label = sample["label"]

            # 1. Évaluation au niveau du clip (1 clip central)
            clip_tensor = load_video_clips_for_evaluation(video_path, num_clips=1)
            if clip_tensor is None:
                continue
            
            outputs_clip = model(clip_tensor.to(device))
            _, pred_clip = torch.max(outputs_clip, dim=1)
            
            # 2. Évaluation au niveau de la vidéo (5 clips moyennés)
            multi_clips = load_video_clips_for_evaluation(video_path, num_clips=5)
            outputs_multi = model(multi_clips.to(device))
            # Moyenne des sorties softmax / logits sur les 5 clips
            avg_output = torch.mean(outputs_multi, dim=0, keepdim=True)
            _, pred_video = torch.max(avg_output, dim=1)

            y_true.append(label)
            clip_preds.append(pred_clip.item())
            video_preds.append(pred_video.item())

            if (i + 1) % 100 == 0:
                print(f"Traitement de {i + 1}/{len(test_dataset.samples)} vidéos de test...")

    # Calcul des métriques globales
    y_true = np.array(y_true)
    clip_preds = np.array(clip_preds)
    video_preds = np.array(video_preds)

    clip_acc = np.mean(clip_preds == y_true)
    video_acc = np.mean(video_preds == y_true)

    print(f"\n--- Résultats d'évaluation ---")
    print(f"Précision au niveau du clip (1 clip) : {clip_acc * 100:.2f}%")
    print(f"Précision au niveau de la vidéo (5 clips) : {video_acc * 100:.2f}%")

    return y_true, video_preds, test_dataset.classes


def plot_training_curves(log_csv_path: str, output_plot_path: str = "./training_curves.png"):
    """
    Trace les courbes de perte et d'exactitude à partir du journal CSV d'entraînement.
    """
    epochs, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []

    if not os.path.exists(log_csv_path):
        print(f"Fichier de journalisation introuvable : {log_csv_path}")
        return

    with open(log_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row["train_acc"]) * 100)
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_acc"]) * 100)

    plt.figure(figsize=(12, 5))

    # Graphique des pertes (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Entraînement Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Époques")
    plt.ylabel("Perte (Loss)")
    plt.title("Évolution de la perte par époque")
    plt.legend()
    plt.grid(True)

    # Graphique de la précision (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Entraînement Acc (%)")
    plt.plot(epochs, val_acc, label="Validation Acc (%)")
    plt.xlabel("Époques")
    plt.ylabel("Exactitude (%)")
    plt.title("Évolution de la précision par époque")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_plot_path)
    print(f"Courbes d'apprentissage sauvegardées dans : {output_plot_path}")
    plt.close()


if __name__ == "__main__":
    MANIFEST_CSV = "./ucf101_manifest.csv"
    CHECKPOINT_3D = "./checkpoints/best_plain_3d.pth"
    LOG_CSV_3D = "./checkpoints/training_log_3d.csv"

    if os.path.exists(MANIFEST_CSV):
        # 1. Évaluation du modèle 3D
        y_true, y_pred, classes = evaluate_model(MANIFEST_CSV, CHECKPOINT_3D, temporal=True)
        
        # 2. Tracé des courbes d'apprentissage[cite: 4]
        plot_training_curves(LOG_CSV_3D, output_plot_path="./curves_plain3d.png")
    else:
        print(f"Manifeste non trouvé : {MANIFEST_CSV}. Exécutez d'abord extract_frames.py.")
