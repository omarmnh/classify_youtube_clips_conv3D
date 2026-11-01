"""
train.py
Script d'entraînement pour le modèle 3D CNN sur UCF101 (Lab 5).
Gère l'optimisation SGD, le gradient clipping, la validation et la journalisation CSV.
"""

import os
import csv
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from model3d import VideoCNN3D
from data_loader import get_data_loaders


def train_model(
    manifest_csv: str,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 0.025,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    grad_clip_norm: float = 5.0,
    output_dir: str = "./checkpoints",
    temporal: bool = True
):
    """
    Entraîne le modèle 3D CNN sur UCF101 et enregistre le meilleur point de contrôle.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du périphérique : {device}")

    # 1. Chargement des données
    print("Initialisation des chargeurs de données...")
    train_loader, val_loader, num_classes = get_data_loaders(
        manifest_csv=manifest_csv, batch_size=batch_size, num_workers=4
    )

    # 2. Initialisation du modèle
    print(f"Création du modèle 3D CNN (classes = {num_classes}, temporel = {temporal})...")
    model = VideoCNN3D(num_classes=num_classes, dropout=0.5, temporal=temporal).to(device)

    # 3. Perte et Optimiseur (spécifications du TP : SGD, momentum 0.9, weight decay 5e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing optionnel recommandé
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # Scheduler optionnel pour réduire le taux d'apprentissage si nécessaire
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Fichiers de journalisation et de sauvegarde
    model_suffix = "_3d" if temporal else "_2d"
    best_model_path = os.path.join(output_dir, f"best_plain{model_suffix}.pth")
    log_csv_path = os.path.join(output_dir, f"training_log{model_suffix}.csv")

    # Initialisation du journal CSV
    with open(log_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    best_val_acc = 0.0
    print(f"Début de l'entraînement pour {epochs} époques...")

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        # --- PHASE D'ENTRAÎNEMENT ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for clips, labels in train_loader:
            clips = clips.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            
            loss.backward()

            # Écrêtage des gradients pour stabiliser l'entraînement 3D
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()

            running_loss += loss.item() * clips.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # --- PHASE DE VALIDATION ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for clips, labels in val_loader:
                clips = clips.to(device)
                labels = labels.to(device)

                outputs = model(clips)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * clips.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        # Mise à jour du taux d'apprentissage
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        epoch_duration = time.time() - epoch_start_time

        # Affichage des métriques de l'époque
        print(
            f"Époque [{epoch}/{epochs}] ({epoch_duration:.1f}s) | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}% | "
            f"LR: {current_lr:.5f}"
        )

        # Enregistrement dans le fichier CSV de log[cite: 4]
        with open(log_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc, current_lr])

        # Sauvegarde du meilleur modèle basé sur la précision de validation[cite: 4]
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
            }, best_model_path)
            print(f"--> Nouveau meilleur modèle sauvegardé avec une précision de validation de {best_val_acc * 100:.2f}% (-> {best_model_path})")

    print(f"Entraînement terminé. Meilleure précision de validation : {best_val_acc * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement du modèle 3D CNN sur UCF101")
    parser.add_argument("--manifest", type=str, default="./ucf101_manifest.csv", help="Chemin vers le fichier manifeste CSV")
    parser.add_argument("--epochs", type=int, default=60, help="Nombre d'époques d'entraînement (min 60)")
    parser.add_argument("--batch_size", type=int, default=32, help="Taille des lots")
    parser.add_argument("--lr", type=float, default=0.025, help="Taux d'apprentissage initial")
    parser.add_argument("--ablate_2d", action="store_true", help="Activer le mode ablation 2D (temporal=False)")

    args = parser.parse_args()

    train_model(
        manifest_csv=args.manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temporal=not args.ablate_2d
    )
