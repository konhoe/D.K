from typing import Literal, Tuple
import os
import torch
import torch.nn as nn
from transformers import CLIPModel


class ClipBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        dtype: Literal["fp32", "bf16", "fp16"] = "fp32",
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,  # 0이면 완전 동결
    ):
        super().__init__()

        # 로컬 디렉토리면 오프라인 로드, 아니면 사전에 TRANSFORMERS_OFFLINE=1 권장
        local_only = os.path.isdir(model_name)
        self.clip: CLIPModel = CLIPModel.from_pretrained(
            model_name, local_files_only=local_only
        )

        # 편의 alias
        self.vision = self.clip.vision_model
        self.visual_projection = self.clip.visual_projection

        # hidden dim / projection dim
        self.hidden_dim = self.vision.config.hidden_size
        self.embed_dim = getattr(self.clip.config, "projection_dim", self.hidden_dim)

        # 버전 차이 대비: position_ids 강제 등록(텍스트/비전)
        self._ensure_position_ids()

        # dtype 지정: 텍스트는 사용 안 하므로 vision/projection만 캐스팅
        if dtype == "bf16":
            cast_dtype = torch.bfloat16
        elif dtype == "fp16":
            cast_dtype = torch.float16
        else:
            cast_dtype = torch.float32

        self.vision.to(dtype=cast_dtype)
        self.visual_projection.to(dtype=cast_dtype)

        # 동결/언프리즈
        if freeze_backbone:
            for p in self.clip.parameters():
                p.requires_grad = False
        if unfreeze_last_n_blocks > 0:
            blocks = list(self.vision.encoder.layers)
            for blk in blocks[-unfreeze_last_n_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True

    # ------------------------------------------------
    # position_ids 관련 유틸
    # ------------------------------------------------
    def _ensure_position_ids(self) -> None:
        """HF/환경 버전에 따라 state_dict에 없을 수 있는 position_ids를 강제로 등록."""
        # 텍스트 position_ids (보통 77)
        try:
            txt_conf = self.clip.text_model.config
            txt_len = int(getattr(txt_conf, "max_position_embeddings", 77))
        except Exception:
            txt_len = 77
        txt_pos = torch.arange(0, txt_len, dtype=torch.long).unsqueeze(0)
        try:
            txt_emb = self.clip.text_model.embeddings
            if not hasattr(txt_emb, "position_ids"):
                txt_emb.register_buffer("position_ids", txt_pos, persistent=True)
        except Exception:
            pass  

        # 비전 position_ids (ViT-L/14: 224/14=16 → 16*16+1=257)
        try:
            vconf = self.clip.vision_model.config
            v_len = (vconf.image_size // vconf.patch_size) ** 2 + 1
        except Exception:
            v_len = 257
        vis_pos = torch.arange(0, v_len, dtype=torch.long).unsqueeze(0)
        try:
            vemb = self.clip.vision_model.embeddings
            if not hasattr(vemb, "position_ids"):
                vemb.register_buffer("position_ids", vis_pos, persistent=True)
        except Exception:
            pass

    def state_dict(self, destination=None, prefix: str = '', keep_vars: bool = False):
        """
        저장 시 'clip.vision_model.embeddings.position_ids' 뿐 아니라
        'vision.embeddings.position_ids' 키도 강제로 포함시켜 strict 로더 호환성 확보.
        """
        sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        src_key = prefix + "clip.vision_model.embeddings.position_ids"
        dst_key = prefix + "vision.embeddings.position_ids"
        if src_key in sd and dst_key not in sd:
            # 동일 텐서를 복제 저장(메모리 오버헤드는 미미)
            sd[dst_key] = sd[src_key]

        # 텍스트/비전 쪽 키가 없으면 여기서도 한 번 더 보강
        txt_key = prefix + "clip.text_model.embeddings.position_ids"
        if txt_key not in sd:
            sd[txt_key] = torch.arange(0, 77, dtype=torch.long).unsqueeze(0)
        vis_key = src_key
        if vis_key not in sd:
            sd[vis_key] = torch.arange(0, 257, dtype=torch.long).unsqueeze(0)

        return sd

    # ------------------------------------------------
    # 기존 이미지 임베딩용 (그대로 유지)
    # ------------------------------------------------
    @torch.no_grad()
    def forward_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        이미지 전용 인퍼런스:
          pixel_values: (B,3,H,W)
          return:      (B, embed_dim)
        """
        dev = next(self.vision.parameters()).device
        dt = next(self.vision.parameters()).dtype
        pixel_values = pixel_values.to(device=dev, dtype=dt)

        out = self.vision(pixel_values=pixel_values)
        pooled = (
            out.pooler_output
            if hasattr(out, "pooler_output") and out.pooler_output is not None
            else out.last_hidden_state[:, 0, :]
        )
        emb = self.visual_projection(pooled)
        return emb


    def forward_tokens(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        dev = next(self.vision.parameters()).device
        dt = next(self.vision.parameters()).dtype

        if pixel_values.ndim == 4:
            # (B,3,H,W) → T=1로 처리
            B, C, H, W = pixel_values.shape
            T = 1
            x = pixel_values
        elif pixel_values.ndim == 5:
            # (B,T,3,H,W) → (B*T,3,H,W)
            B, T, C, H, W = pixel_values.shape
            x = pixel_values.reshape(B * T, C, H, W)
        else:
            raise ValueError(f"pixel_values.ndim must be 4 or 5, got {pixel_values.ndim}")

        x = x.to(device=dev, dtype=dt)

        out = self.vision(pixel_values=x)
        # last_hidden_state: (B*T, N, C_hidden)
        hidden = out.last_hidden_state  # (BT, N, C)
        # (BT, N, C) -> (N, BT, C)
        tokens = hidden.transpose(0, 1).contiguous()

        return tokens, B, T

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        기본 forward는 토큰 시퀀스를 반환:
          return: (tokens, B, T)
          - tokens: (N, BT, C_hidden)
        DeepfakeModel + Adapter에서 이걸 받아서 D2ST-Adapter 통과시키고,
        CLS pooling → classifier에 넣으면 됨.
        """
        return self.forward_tokens(pixel_values)
