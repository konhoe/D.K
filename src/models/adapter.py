from __future__ import annotations
import math
import torch
import torch.nn as nn
from einops import rearrange

from .attention import ViT_DeformAttention


class ViT_D2ST_Adapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.args = cfg

        self.in_channels = cfg.ADAPTER.WIDTH
        self.out_channels = cfg.ADAPTER.WIDTH
        self.adapter_channels = int(
            cfg.ADAPTER.WIDTH * cfg.ADAPTER.ADAPTER_SCALE
        )

        self.down = nn.Linear(self.in_channels, self.adapter_channels)
        self.gelu1 = nn.GELU()

        self.pos_embed = nn.Conv3d(
            in_channels=self.adapter_channels,
            out_channels=self.adapter_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            groups=self.adapter_channels,
        )

        heads = getattr(cfg.ADAPTER, "HEADS", 4)
        groups = getattr(cfg.ADAPTER, "GROUPS", 4)

        # 🔥 여기서 T에 따라 spatial kernel을 바꿔줌
        T = cfg.DATA.NUM_INPUT_FRAMES
        if T <= 1:
            # 이미지 모드: time 축은 1이므로 kernel_t=1만 허용
            spatial_kernel = (1, 5, 5)
            spatial_stride = (1, 3, 3)
        else:
            # 비디오 모드: 논문 값 그대로
            spatial_kernel = (4, 5, 5)
            spatial_stride = (4, 3, 3)

        self.s_ln = nn.LayerNorm(self.adapter_channels)
        self.s_attn = ViT_DeformAttention(
            cfg=cfg,
            dim=self.adapter_channels,
            heads=heads,
            groups=groups,
            kernel_size=spatial_kernel,
            stride=spatial_stride,
            padding=(0, 0, 0),
        )

        self.t_ln = nn.LayerNorm(self.adapter_channels)
        self.t_attn = ViT_DeformAttention(
            cfg=cfg,
            dim=self.adapter_channels,
            heads=heads,
            groups=groups,
            kernel_size=(1, 7, 7),  # time 축은 원래부터 1이라 여기엔 문제 없음
            stride=(1, 7, 7),
            padding=(0, 0, 0),
        )

        self.gelu = nn.GELU()
        self.up = nn.Linear(self.adapter_channels, self.out_channels)
        self.gelu2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, BT, C_in)
           N  = H*W + 1 (CLS 포함)
           BT = B*T
        return: (N, BT, C_in)
        """
        n, bt, c = x.shape
        x_in = x

        # H, T 복원
        T = self.args.DATA.NUM_INPUT_FRAMES
        if bt % T != 0:
            raise ValueError(
                f"[ViT_D2ST_Adapter] BT({bt})가 T({T})로 나누어 떨어지지 않음. "
                f"B가 정수가 되도록 (B,T) 구성을 맞춰줘야 함."
            )
        B = bt // T
        # CLS 제외한 토큰 개수 = H*W
        H = round(math.sqrt(n - 1))
        if H * H != (n - 1):
            # 패치 수가 정사각형이 아니면 여기에서 에러를 보는 게 디버깅에 도움 됨
            raise ValueError(
                f"[ViT_D2ST_Adapter] n-1={n-1}이 완전제곱이 아님. "
                f"H^2 = n-1이 되어야 하는데, H={H}일 때 {H*H}."
            )

        # -------------------------
        # 1. Down-projection (token-wise)
        # -------------------------
        x = self.down(x)      # (N, BT, C')
        x = self.gelu1(x)     # (N, BT, C')

        # -------------------------
        # 2. CLS 분리
        # -------------------------
        cls = x[0:1, :, :]    # (1, BT, C')
        x_spatial = x[1:, :, :]  # (H*W, BT, C')

        # -------------------------
        # 3. 5D Positional Embedding (B, C', T, H, W)
        # -------------------------
        # (H*W, BT, C') -> (B, C', T, H, W)
        x_spatial = rearrange(
            x_spatial,
            '(h w) (b t) c -> b c t h w',
            h=H,
            t=T,
        )
        x_spatial = x_spatial + self.pos_embed(x_spatial)

        # 다시 토큰 시퀀스로
        x_spatial = rearrange(
            x_spatial,
            'b c t h w -> (h w) (b t) c',
        )

        # CLS 다시 붙이기 → (N, BT, C')
        x = torch.cat([cls, x_spatial], dim=0)

        # -------------------------
        # 4. Spatial / Temporal Deformable Attention
        # -------------------------
        xs = x + self.s_attn(self.s_ln(x))   # spatial branch
        xt = x + self.t_attn(self.t_ln(x))   # temporal branch

        # 두 branch를 평균으로 fuse
        x = 0.5 * (xs + xt)
        x = self.gelu(x)

        # -------------------------
        # 5. Up-projection + Residual
        # -------------------------
        x = self.up(x)      # (N, BT, C_in)
        x = self.gelu2(x)
        x = x + x_in        # residual

        return x


__all__ = [
    "ViT_D2ST_Adapter",
]
