from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageAdapter(nn.Module):
    """
    이미지 전용 Adapter
    - 입력: (B, D)
    - 출력: (B, D)
    - 역할: CLIP image feature를 deepfake task에 맞게 살짝 비틀어주는 bottleneck MLP + residual
    """
    def __init__(
        self,
        dim: int,
        bottleneck: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.bottleneck = bottleneck

        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D)
        return: (B, D)
        """
        residual = x
        x = self.norm(x)
        x = self.down(x)       # (B, r)
        x = self.act(x)
        x = self.up(x)         # (B, D)
        x = self.dropout(x)
        return x + residual    # (B, D)


class TemporalAdapter(nn.Module):
    """
    비디오 전용 Temporal Adapter
    - 입력: (B, T, D)
    - 출력: (B, T, D)
    - 역할: time 축(T)에 대해 1D conv로 temporal 패턴 잡기
    - ST-Adapter의 temporal branch 느낌
    """
    def __init__(
        self,
        dim: int,
        bottleneck: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.bottleneck = bottleneck

        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)

        padding = kernel_size // 2
        # Conv1d: (B, r, T) -> (B, r, T)
        self.temporal_conv = nn.Conv1d(
            in_channels=bottleneck,
            out_channels=bottleneck,
            kernel_size=kernel_size,
            padding=padding,
            groups=1,
        )

        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        return: (B, T, D)
        """
        residual = x
        B, T, D = x.shape

        x = self.norm(x)
        x = self.down(x)               # (B, T, r)

        # Conv1d는 (B, C, T) 포맷 → (B, r, T)로 transpose
        x = x.transpose(1, 2)          # (B, r, T)
        x = self.temporal_conv(x)      # (B, r, T)
        x = x.transpose(1, 2)          # (B, T, r)

        x = self.act(x)
        x = self.up(x)                 # (B, T, D)
        x = self.dropout(x)

        return x + residual            # (B, T, D)


class VideoAdapter(nn.Module):
    """
    비디오 전용 Adapter
    - 입력: (B, T, D)
    - 출력: (B, T, D)
    - 구조:
        1) frame-wise ImageAdapter (spatial)
        2) TemporalAdapter (temporal)
      둘 다 residual로 쌓아서 ST-Adapter 비슷한 효과
    """
    def __init__(
        self,
        dim: int,
        bottleneck: int = 256,
        dropout: float = 0.1,
        kernel_size: int = 3,
        n_layers_spatial: int = 1,
        n_layers_temporal: int = 1,
    ):
        super().__init__()
        self.dim = dim

        # frame-wise spatial adapter (ImageAdapter를 T프레임에 공유해서 적용)
        self.spatial_layers = nn.ModuleList([
            ImageAdapter(dim=dim, bottleneck=bottleneck, dropout=dropout)
            for _ in range(n_layers_spatial)
        ])

        # temporal adapter stack
        self.temporal_layers = nn.ModuleList([
            TemporalAdapter(
                dim=dim,
                bottleneck=bottleneck,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            for _ in range(n_layers_temporal)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        return: (B, T, D)
        """
        B, T, D = x.shape

        # 1) frame-wise spatial adapter
        # (B,T,D) -> (B*T,D)로 펼쳐서 ImageAdapter 여러 층 태움
        x_2d = x.view(B * T, D)          # (B*T, D)
        for layer in self.spatial_layers:
            x_2d = layer(x_2d)           # (B*T, D)
        x = x_2d.view(B, T, D)           # (B, T, D)

        # 2) temporal adapter
        for t_layer in self.temporal_layers:
            x = t_layer(x)               # (B, T, D)

        return x                         # (B, T, D)


__all__ = [
    "ImageAdapter",
    "TemporalAdapter",
    "VideoAdapter",
]
