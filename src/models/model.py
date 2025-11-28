from __future__ import annotations
import torch
import torch.nn as nn
from typing import Literal, Optional

from .clip_backbone import ClipBackbone
from .adapter import ViT_D2ST_Adapter
from .config import D2STConfig, AdapterConfig, DataConfig


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
        return self.net(x)   # (B, num_classes)


class DeepfakeModel(nn.Module):
    """
    논문 구조 기반:
    - CLIP Vision Backbone → (N, BT, C)
    - D2ST-Adapter → (N, BT, C)
    - CLS token → pooling → classifier
    """
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        *,
        dtype: Literal["fp32", "bf16", "fp16"] = "fp32",
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,
        num_classes: int = 2,
        d2st_num_frames: int = 12,
        d2st_scale: float = 0.25,
        hidden_mult: int = 2,
        temporal_pool: Literal["mean", "max"] = "mean",
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
    ):
        super().__init__()

        # --------------------------------------------------------------
        # 1) CLIP Backbone
        # --------------------------------------------------------------
        self.backbone = ClipBackbone(
            model_name=clip_model_name,
            dtype=dtype,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )
        C = self.backbone.hidden_dim # CLIP hidden dim (=1024 for ViT-L/14)

        # --------------------------------------------------------------
        # 2) D2ST Config
        # --------------------------------------------------------------
        self.cfg = D2STConfig(
            ADAPTER=AdapterConfig(
                WIDTH=C,
                ADAPTER_SCALE=d2st_scale,
                HEADS=4,
                GROUPS=4,
            ),
            DATA=DataConfig(
                NUM_INPUT_FRAMES=d2st_num_frames
            ),
        )

        # --------------------------------------------------------------
        # 3) D2ST-Adapter (논문 핵심)
        # --------------------------------------------------------------
        self.adapter = ViT_D2ST_Adapter(self.cfg)

        # --------------------------------------------------------------
        # 4) Classifier
        # --------------------------------------------------------------
        self.cls_head = MLPHead(
            dim=C,
            num_classes=num_classes,
            hidden_mult=hidden_mult,
        )

        self.temporal_pool = temporal_pool
        self.num_classes = num_classes
        self.id2label = id2label or {i: str(i) for i in range(num_classes)}
        self.label2id = label2id or {v: k for k, v in self.id2label.items()}

    # --------------------------------------------------------------
    # Image forward (Stage-1)
    # --------------------------------------------------------------
    def forward_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: (B,3,H,W)
        """
        tokens, B, T = self.backbone(pixel_values)   # tokens: (N, BT, C), T=1

        # cfg 안 NUM_INPUT_FRAMES를 T에 맞춰주거나,
        # 아예 image용 config에서 1로 세팅해도 됨.
        if self.cfg.DATA.NUM_INPUT_FRAMES != T:
            self.cfg.DATA.NUM_INPUT_FRAMES = T  # 이미지 전용 모델이면 이렇게 덮어도 OK

        tokens = self.adapter(tokens)          # (N, BT, C)

        cls = tokens[0]                        # (BT, C) = (B*1, C)
        cls = cls.view(B, T, -1)[:, 0, :]      # (B, C)

        logits = self.cls_head(cls)            # (B, num_classes)
        return logits

    # --------------------------------------------------------------
    # Video forward (Stage-2)
    # --------------------------------------------------------------
    def forward_video(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: (B,T,3,H,W)
        """
        tokens, B, T = self.backbone(pixel_values)   # (N, BT, C)

        assert T == self.cfg.DATA.NUM_INPUT_FRAMES, \
            f"T={T}, but cfg expects {self.cfg.DATA.NUM_INPUT_FRAMES}"

        tokens = self.adapter(tokens)          # (N, BT, C)

        cls = tokens[0]                        # (BT, C)
        cls = cls.view(B, T, -1)               # (B, T, C)

        if self.temporal_pool == "mean":
            feat = cls.mean(dim=1)             # (B, C)
        elif self.temporal_pool == "max":
            feat = cls.max(dim=1).values
        else:
            raise ValueError(self.temporal_pool)

        logits = self.cls_head(feat)           # (B, num_classes)
        return logits

    # --------------------------------------------------------------
    # Unified forward
    # --------------------------------------------------------------
    def forward(self, pixel_values: torch.Tensor, mode: Optional[str] = None, **kwargs):
        if mode is None:
            if pixel_values.dim() == 4:
                mode = "image"
            elif pixel_values.dim() == 5:
                mode = "video"
            else:
                raise ValueError(f"Unexpected dim={pixel_values.dim()}")

        if mode == "image":
            return self.forward_image(pixel_values)
        elif mode == "video":
            return self.forward_video(pixel_values)
        else:
            raise ValueError(mode)
