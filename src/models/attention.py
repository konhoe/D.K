from __future__ import annotations
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum


class LayerNormProxy(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W) -> (B, T, H, W, C)
        x = rearrange(x, "b c t h w -> b t h w c")
        x = self.norm(x)
        # 다시 (B, C, T, H, W)
        x = rearrange(x, "b t h w c -> b c t h w")
        return x


class ViT_DeformAttention(nn.Module):
    """
    ViT 토큰용 Deformable Spatio-Temporal Attention.

    입력/출력:
        x: (N, BT, C)
           - N  : 토큰 수 (CLS + H*W)
           - B  : batch size
           - T  : frame 수 (cfg.DATA.NUM_INPUT_FRAMES)
           - BT : B*T
           - C  : dim

    내부:
        1) Query 기반 offset 예측 (Conv3D)
        2) grid_sample로 x를 sampling → K, V 생성
        3) Multi-head attention
    """

    def __init__(
        self,
        cfg,
        dim: int,
        heads: int,
        groups: int,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        padding: Tuple[int, int, int],
    ):
        super().__init__()
        self.args = cfg
        self.dim = dim
        self.heads = heads
        self.head_channels = dim // heads
        self.scale = self.head_channels**-0.5

        self.groups = groups
        self.group_channels = self.dim // self.groups
        self.factor = 2.0  # offset 범위 scaling

        # -------------------------
        # offset prediction conv (depthwise Conv3D)
        # -------------------------
        self.conv_offset = nn.Sequential(
            nn.Conv3d(
                in_channels=self.group_channels,
                out_channels=self.group_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=self.group_channels,
            ),
            LayerNormProxy(self.group_channels),
            nn.GELU(),
            nn.Conv3d(
                in_channels=self.group_channels,
                out_channels=3,  # (dz, dy, dx)
                kernel_size=(1, 1, 1),
                bias=False,
            ),
        )

        # -------------------------
        # Q / K / V projection
        # -------------------------
        self.proj_q = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.proj_k = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.proj_v = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.proj_out = nn.Linear(in_features=self.dim, out_features=self.dim)

    @torch.no_grad()
    def _get_ref_points(
        self,
        T: int,
        H: int,
        W: int,
        B: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        3D normalized reference grid 생성.
        출력: (B*groups, T, H, W, 3), 값 범위 [-1,1]
        """

        # meshgrid: z(시간), y, x
        ref_z, ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, T - 0.5, T, dtype=dtype, device=device),
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
            indexing="ij",  # 명시적으로 설정
        )
        # (T,H,W,3)
        ref = torch.stack((ref_z, ref_y, ref_x), dim=-1)

        # [-1,1] 범위로 정규화
        ref[..., 0].div_(T).mul_(2).sub_(1)  # z
        ref[..., 1].div_(H).mul_(2).sub_(1)  # y
        ref[..., 2].div_(W).mul_(2).sub_(1)  # x

        # (B*groups, T, H, W, 3)
        ref = ref[None, ...].expand(B * self.groups, -1, -1, -1, -1)
        return ref

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, BT, C)
        return: (N, BT, C)
        """
        n, BT, C = x.shape
        T = self.args.DATA.NUM_INPUT_FRAMES
        if BT % T != 0:
            raise ValueError(
                f"[ViT_DeformAttention] BT({BT})가 T({T})로 나누어 떨어지지 않음."
            )
        B = BT // T
        H = round(math.sqrt(n - 1))  # 패치 수 = H*W, CLS 제외
        if H * H != (n - 1):
            raise ValueError(
                f"[ViT_DeformAttention] n-1={n-1}이 완전제곱이 아님. "
                f"H^2 = n-1이어야 하는데 H={H}, H^2={H*H}."
            )

        dtype, device = x.dtype, x.device

        # -------------------------
        # 1. Q projection
        # -------------------------
        q = self.proj_q(x)  # (N, BT, C)

        # offset 계산용 feature: spatial token만 사용 (CLS 제외)
        # q[1:, :, :] → (H*W, B*T, C)
        q_off = rearrange(
            q[1:, :, :],
            "(h w) (b t) c -> b c t h w",
            h=H,
            t=T,
        )  # (B, C, T, H, W)

        # group-wise로 나누기
        q_off = rearrange(
            q_off,
            "b (g c) t h w -> (b g) c t h w",
            g=self.groups,
            c=self.group_channels,
        )  # (B*G, C_group, T, H, W)

        # -------------------------
        # 2. offset prediction (Conv3D)
        # -------------------------
        offset = self.conv_offset(q_off)  # (B*G, 3, T', H', W')
        Tp, Hp, Wp = offset.size(2), offset.size(3), offset.size(4)
        # n_sample = Tp * Hp * Wp  # 필요하면 사용

        # offset 범위 제한
        offset_range = torch.tensor(
            [
                min(1.0, self.factor / Tp),
                min(1.0, self.factor / Hp),
                min(1.0, self.factor / Wp),
            ],
            device=device,
            dtype=dtype,
        ).view(1, 3, 1, 1, 1)
        offset = offset.tanh().mul(offset_range)  # (B*G, 3, T', H', W')
        offset = rearrange(offset, "b p t h w -> b t h w p")  # (B*G, T', H', W', 3)

        # -------------------------
        # 3. reference grid + offset → sampling grid
        # -------------------------
        reference = self._get_ref_points(
            T=Tp,
            H=Hp,
            W=Wp,
            B=B,
            dtype=dtype,
            device=device,
        )  # (B*G, T', H', W', 3)
        pos = offset + reference  # (B*G, T', H', W', 3), [-1,1] 범위

        # -------------------------
        # 4. 원본 feature에서 sampling (grid_sample)
        # -------------------------
        # x[1:, :, :] : (H*W, B*T, C) -> (B, C, T, H, W)
        x_sampled = rearrange(
            x[1:, :, :],
            "(h w) (b t) c -> b c t h w",
            h=H,
            t=T,
        )  # (B, C, T, H, W)

        # group-wise로 펼치기: (B*G, C_group, T, H, W)
        x_sampled = rearrange(
            x_sampled,
            "b (g c) t h w -> (b g) c t h w",
            g=self.groups,
            c=self.group_channels,
        )

        # grid_sample: grid는 (z,y,x) → (x,y,z) 순으로 넣어야 함
        x_sampled = F.grid_sample(
            input=x_sampled,
            grid=pos[..., (2, 1, 0)],  # (..., x, y, z)
            mode="bilinear",
            align_corners=True,
        )  # (B*G, C_group, T', H', W')

        # 다시 (B, C, T', H', W')
        x_sampled = rearrange(
            x_sampled,
            "(b g) c t h w -> b (g c) t h w",
            g=self.groups,
        )  # (B, C, T', H', W')

        # (B, C, T', H', W') -> (B, T'*H'*W', C)
        x_sampled = rearrange(
            x_sampled,
            "b c t h w -> b (t h w) c",
        )  # (B, N_sample, C)

        # -------------------------
        # 5. Q / K / V 만들기 (multi-head)
        # -------------------------

        # Q: (N, BT, C) -> (B, C, T*N)
        q_proj = rearrange(
            q,
            "n (b t) c -> b c (t n)",
            b=B,
        )  # (B, C, T*N)
        # head 축으로 나누기
        q_proj = rearrange(
            q_proj,
            "b (h c) n -> (b h) c n",
            h=self.heads,
        )  # (B*H, C_head, T*N)

        # K, V는 sampled feature에서
        k = self.proj_k(x_sampled)  # (B, N_sample, C)
        v = self.proj_v(x_sampled)  # (B, N_sample, C)

        k = rearrange(
            k,
            "b n (h c) -> (b h) c n",
            h=self.heads,
        )  # (B*H, C_head, N_sample)
        v = rearrange(
            v,
            "b n (h c) -> (b h) c n",
            h=self.heads,
        )  # (B*H, C_head, N_sample)

        # -------------------------
        # 6. Attention
        # -------------------------
        # attn: (B*H, T*N, N_sample)
        attn = einsum("b c m, b c n -> b m n", q_proj, k)  # Q·K^T
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)

        # out: (B*H, C_head, T*N)
        out = einsum("b m n, b c n -> b c m", attn, v)

        # (B*H, C_head, T*N) -> (B, C, T*N)
        out = rearrange(
            out,
            "(b h) c n -> b (h c) n",
            h=self.heads,
        )  # (B, C, T*N)

        # (B, C, T*N) -> (N, BT, C)
        out = rearrange(
            out,
            "b c (t n) -> n (b t) c",
            t=T,
        )  # (N, BT, C)

        out = self.proj_out(out)  # (N, BT, C)
        return out


__all__ = [
    "LayerNormProxy",
    "ViT_DeformAttention",
]
