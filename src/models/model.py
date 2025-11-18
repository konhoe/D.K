from __future__ import annotations
import torch
import torch.nn as nn
from typing import Literal, Optional

from .clip_backbone import ClipBackbone
from .adapter import ImageAdapter, VideoAdapter


class MLPHead(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int = 2,
        hidden_mult: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden = dim * hidden_mult
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        return self.net(x)   # (B, num_classes)


class DeepfakeModel(nn.Module):
    """
    - CLIP 비전 백본 공유
    - 이미지: (B,3,H,W) → (B,D) → ImageAdapter → ImageHead
    - 비디오: (B,T,3,H,W) → (B,T,D) → VideoAdapter → T-pool → VideoHead
    """
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        *,
        dtype: Literal["fp32", "bf16", "fp16"] = "fp32",
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,
        bottleneck: int = 256,
        num_classes: int = 2,
        hidden_mult: int = 2,
        temporal_pool: Literal["mean", "max"] = "mean",
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
    ):
        super().__init__()

        # 1) CLIP backbone
        self.backbone = ClipBackbone(
            model_name=clip_model_name,
            dtype=dtype,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )
        D = self.backbone.embed_dim

        # 2) Adapters
        self.image_adapter = ImageAdapter(
            dim=D,
            bottleneck=bottleneck,
            dropout=0.1,
        )

        self.video_adapter = VideoAdapter(
            dim=D,
            bottleneck=bottleneck,
            dropout=0.1,
            kernel_size=3,
            n_layers_spatial=1,
            n_layers_temporal=2,
        )

        # 3) Heads
        self.image_head = MLPHead(
            dim=D,
            num_classes=num_classes,
            hidden_mult=hidden_mult,
            dropout=0.1,
        )
        self.video_head = MLPHead(
            dim=D,
            num_classes=num_classes,
            hidden_mult=hidden_mult,
            dropout=0.1,
        )

        # 기타 메타 정보
        self.temporal_pool = temporal_pool
        self.num_classes = num_classes
        self.id2label = id2label or {i: str(i) for i in range(num_classes)}
        self.label2id = label2id or {v: k for k, v in self.id2label.items()}

    # -------------------------
    # 1) 이미지용 forward
    # -------------------------
    def forward_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: (B,3,H,W)
        return: logits (B, num_classes)
        """
        # CLIP feature
        feat = self.backbone.forward_images(pixel_values)   # (B,D)

        # Image adapter
        feat = self.image_adapter(feat)                     # (B,D)

        # Head
        logits = self.image_head(feat)                      # (B,num_classes)
        return logits

    # -------------------------
    # 2) 비디오용 forward
    # -------------------------
    def forward_video(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: (B,T,3,H,W)
        return: logits (B, num_classes)
        """
        B, T, C, H, W = pixel_values.shape

        # (B,T,3,H,W) -> (B*T,3,H,W)
        x_flat = pixel_values.view(B * T, C, H, W)

        # CLIP feature
        feat_flat = self.backbone.forward_images(x_flat)    # (B*T, D)

        # (B*T,D) -> (B,T,D)
        D = feat_flat.shape[-1]
        feat = feat_flat.view(B, T, D)                      # (B,T,D)

        # Video adapter (spatial + temporal)
        feat = self.video_adapter(feat)                     # (B,T,D)

        # Temporal pooling
        if self.temporal_pool == "mean":
            feat_pooled = feat.mean(dim=1)                  # (B,D)
        elif self.temporal_pool == "max":
            feat_pooled, _ = feat.max(dim=1)                # (B,D)
        else:
            raise ValueError(f"unknown temporal_pool={self.temporal_pool}")

        # Head
        logits = self.video_head(feat_pooled)               # (B,num_classes)
        return logits

    # -------------------------
    # 3) 통합 forward (mode 없으면 dim 보고 자동 분기)
    # -------------------------
    def forward(
        self,
        pixel_values: torch.Tensor = None,
        labels: torch.Tensor = None,   # HF Trainer가 넘겨줘도 받기만 하고 안 씀
        mode: Optional[Literal["image", "video"]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        pixel_values:
          - 이미지 배치: (B,3,H,W)
          - 비디오 배치: (B,T,3,H,W)

        labels:
          - Trainer가 prediction_step / training_step에서 같이 넘길 수 있음.
            여기서는 loss를 직접 계산하지 않으니까 단순히 무시.

        mode:
          - None이면 pixel_values.dim() 기준으로 자동 선택
        """
        if pixel_values is None:
            # 혹시 kwargs로 들어온 경우까지 방어
            pixel_values = kwargs.get("pixel_values", None)
        if pixel_values is None:
            raise ValueError("pixel_values must be provided to DeepfakeModel.forward")

        # 모드 자동 결정
        if mode is None:
            if pixel_values.dim() == 4:
                mode = "image"
            elif pixel_values.dim() == 5:
                mode = "video"
            else:
                raise ValueError(f"Unexpected pixel_values dim={pixel_values.dim()}")

        if mode == "image":
            return self.forward_image(pixel_values)
        elif mode == "video":
            return self.forward_video(pixel_values)
        else:
            raise ValueError(f"unknown mode: {mode}")
