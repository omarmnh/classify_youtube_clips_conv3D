"""
model3d.py
Plain 3D CNN Architecture for Video Action Classification (UCF101)
Implements Conv3DBlock and VideoCNN3D with support for temporal ablation (2D vs 3D).
"""

import torch
import torch.nn as nn


class Conv3DBlock(nn.Module):
    """A standard 3D convolution block consisting of:
    Conv3D -> BatchNorm3D -> ReLU
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple = (3, 3, 3),
        stride: tuple = (1, 1, 1),
        padding: tuple = (1, 1, 1),
        temporal: bool = True
    ):
        super(Conv3DBlock, self).__init__()
        
        # If temporal is False, disable temporal convolution by setting temporal kernel size to 1 and padding to 0
        if not temporal:
            kernel_size = (1, kernel_size[1], kernel_size[2])
            padding = (0, padding[1], padding[2])

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False  # Bias is redundant when followed immediately by BatchNorm3D
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VideoCNN3D(nn.Module):
    """Plain 3D CNN architecture for UCF101 video action recognition.
    Contains 6 convolutional blocks, max pooling layers, global average pooling,
    dropout, and a final classification head.
    """
    def __init__(self, num_classes: int = 101, dropout: float = 0.5, temporal: bool = True):
        super(VideoCNN3D, self).__init__()
        
        self.temporal = temporal

        # Feature Extractor (Table 1 layout)
        self.features = nn.Sequential(
            # Layer 1: Conv3D + BN + ReLU (64 filters)
            Conv3DBlock(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), temporal=self.temporal),
            
            # Layer 2: MaxPool3D
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            # Layer 3: Conv3D + BN + ReLU (128 filters)
            Conv3DBlock(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), temporal=self.temporal),
            
            # Layer 4: MaxPool3D
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            # Layer 5: Conv3D + BN + ReLU (256 filters)
            Conv3DBlock(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), temporal=self.temporal),
            
            # Layer 6: Conv3D + BN + ReLU (256 filters)
            Conv3DBlock(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), temporal=self.temporal),
            
            # Layer 7: MaxPool3D
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            # Layer 8: Conv3D + BN + ReLU (512 filters)
            Conv3DBlock(256, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), temporal=self.temporal),
            
            # Layer 9: Conv3D + BN + ReLU (512 filters)
            Conv3DBlock(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), temporal=self.temporal),
        )

        # Classification Head
        # Global Average Pooling over Time, Height, and Width dimensions (dim 2, 3, 4)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (B, C, T, H, W) e.g., (32, 3, 16, 64, 64)
        """
        x = self.features(x)
        x = self.global_pool(x)       # Shape: (B, 512, 1, 1, 1)
        x = torch.flatten(x, 1)       # Shape: (B, 512)
        x = self.dropout(x)
        x = self.fc(x)                # Shape: (B, num_classes)
        return x


if __name__ == "__main__":
    # Quick sanity check and shape test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dummy batch matching input requirement: Batch=2, Channels=3, Frames=16, Height=64, Width=64
    dummy_input = torch.randn(2, 3, 16, 64, 64).to(device)

    print("\n--- Testing 3D CNN Baseline ---")
    model_3d = VideoCNN3D(num_classes=101, temporal=True).to(device)
    output_3d = model_3d(dummy_input)
    print(f"Output shape (3D): {output_3d.shape}")  # Expected: torch.Size([2, 101])

    print("\n--- Testing 2D Ablation Model (temporal=False) ---")
    model_2d = VideoCNN3D(num_classes=101, temporal=False).to(device)
    output_2d = model_2d(dummy_input)
    print(f"Output shape (2D ablation): {output_2d.shape}")  # Expected: torch.Size([2, 101])
    
    # Parameter count check
    params_3d = sum(p.numel() for p in model_3d.parameters() if p.requires_grad)
    print(f"\nTotal Trainable Parameters: {params_3d:,}")