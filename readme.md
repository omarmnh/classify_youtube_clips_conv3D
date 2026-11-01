# Video Action Classification with a 3D CNN (UCF101)

This repository contains a PyTorch implementation of a 3D Convolutional Neural Network (3D CNN) built from scratch for video action classification on the **UCF101** dataset. The project also includes an ablation study comparing the 3D spatio-temporal model against a per-frame 2D baseline.

---

## 📌 Features & Highlights

- **Video Processing Pipeline**: Frame extraction, downscaling to 80px short-side, and storage as `uint8` `.npy` arrays for fast disk I/O during training.
- **Group-aware Data Splitting**: Splitting by UCF101 group ID (e.g., `g01`, `g02`) instead of individual clips to avoid data leakage and over-optimistic evaluation.
- **Custom 3D CNN Architecture**: ~13.5M parameters, built using custom `Conv3DBlock` (Conv3D + BatchNorm3D + ReLU) modules and Global Average Pooling.
- **Temporal Ablation**: Configurable `temporal` flag switching kernels from $(3 \times k_h \times k_w)$ to $(1 \times k_h \times k_w)$ to evaluate the impact of time modeling.
- **Multi-Clip Video-Level Evaluation**: Aggregates predictions across 5 temporal clips per test video for standard UCF101 evaluation protocol.

---

## 📂 Project Structure

```text
.
├── extract_frames.py    # Decodes .avi videos, samples 16 frames, saves .npy & CSV manifest
├── model3d.py           # VideoCNN3D architecture & Conv3DBlock implementation
├── data_loader.py       # UCF101 PyTorch Dataset, video augmentations & DataLoaders
├── train.py             # Training loop, SGD optimizer, gradient clipping & CSV logging
├── evaluate.py          # Clip-level & Video-level evaluation, learning curves
├── requirements.txt     # Python project dependencies
└── README.md            # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ucf101-3dcnn.git
   cd ucf101-3dcnn
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Pipeline

### 1. Preprocessing (`extract_frames.py`)
Extract 16 evenly spaced frames per video, resize them to $80\text{px}$ short-side, save as `.npy`, and construct the `ucf101_manifest.csv`:
```bash
python extract_frames.py
```

### 2. Training (`train.py`)
Train the plain 3D CNN model with SGD optimizer, momentum $0.9$, weight decay $5 \times 10^{-4}$, and gradient clipping:

* **Train Baseline 3D CNN:**
  ```bash
  python train.py --manifest ./ucf101_manifest.csv --epochs 60 --batch_size 32 --lr 0.025
  ```

* **Train 2D Ablation Model (Per-frame):**
  ```bash
  python train.py --manifest ./ucf101_manifest.csv --epochs 60 --ablate_2d
  ```

### 3. Evaluation (`evaluate.py`)
Evaluate the trained models on the test set using both single-clip and 5-clip averaging protocols:
```bash
python evaluate.py
```

---

## 📊 Expected Performance Benchmark

| Model Variant | Temporal Kernel | Single Clip Acc (%) | 5-Clip Video Acc (%) |
| :--- | :---: | :---: | :---: |
| **3D CNN Baseline** | $3 \times k_h \times k_w$ | ~35 - 45% | **40 - 55%** |
| **2D CNN Ablation** | $1 \times k_h \times k_w$ | ~20 - 30% | ~25 - 35% |

---

## 📜 License
Developed for Lab 5: Video Action Classification with a 3D CNN (Big Data & AI).