# src/clip_prompt.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_backbone import ClipBackbone

class PromptClassifier(nn.Module):
    """
    - CLIP vision backbone은 freeze (이미지 임베딩 추출용)
    - 각 클래스마다 trainable prototype 벡터를 두고
      cosine similarity 기반으로 real/fake 분류하는 구조
    - prompt learning의 'class-specific weight' 버전이라고 보면 됨.
    """
    def __init__(
        self,
        backbone: ClipBackbone,
        num_classes: int,
        init_scale: float = 0.02,
        learn_logit_scale: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim

        # 클래스별 프로토타입 (prompt embedding처럼 동작)
        self.class_embed = nn.Parameter(
            init_scale * torch.randn(num_classes, self.embed_dim)
        )

        # CLIP의 logit_scale 비슷한 learnable temperature
        if learn_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))
        else:
            self.register_buffer("logit_scale", torch.log(torch.tensor(1/0.07)))

        # backbone freeze
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, pixel_values: torch.Tensor):
        """
        pixel_values: (B, 3, H, W)
        return: logits (B, num_classes)
        """
        # 1) 이미지 임베딩
        with torch.no_grad():
            img_emb = self.backbone.forward_images(pixel_values)  # (B, D)

        img_emb = F.normalize(img_emb, dim=-1)           # (B, D)
        cls_emb = F.normalize(self.class_embed, dim=-1)  # (C, D)

        # 2) cosine similarity
        logits = img_emb @ cls_emb.t()                  # (B, C)
        logits = logits * self.logit_scale.exp()
        return logits
