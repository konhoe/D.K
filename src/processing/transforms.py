from __future__ import annotations
import torch

def collate_fn(examples, stage: str = "stage2", video_frames: int = 12):
    """
    stage: "stage1" | "stage2"
    
    examples[i]["pixel_values"]:
      - 이미지: (3,H,W)
      - 비디오 Stage1: (1,3,H,W)
      - 비디오 Stage2: (T,3,H,W)

    반환:
      Stage1 → (B,3,H,W)
      Stage2 → (B,T,3,H,W)
    """

    xs = [ex["pixel_values"] for ex in examples]
    ys = torch.tensor([int(ex["label"]) for ex in examples], dtype=torch.long)

    # --------------------------
    # Stage 1 (single-frame)
    # --------------------------
    if stage == "stage1":
        processed = []
        for t in xs:
            if t.ndim == 4:      # (1,3,H,W) → (3,H,W)
                t = t[0]
            # t is now (3,H,W)
            processed.append(t)

        x = torch.stack(processed, dim=0)   # (B,3,H,W)
        return {"pixel_values": x, "labels": ys}

    # --------------------------
    # Stage 2 (video/full T)
    # --------------------------
    # 이미지/비디오 섞여 있을 때 T 맞춰주기
    proc = []
    T_max = video_frames

    for t in xs:
        if t.ndim == 3:   # 이미지 (3,H,W)
            # repeat to (T_max,3,H,W)
            t = t.unsqueeze(0).repeat(T_max, 1, 1, 1)

        elif t.ndim == 4:
            # 비디오 프레임 수가 부족한 경우 pad
            if t.shape[0] < T_max:
                pad = t[-1:].repeat(T_max - t.shape[0], 1, 1, 1)
                t = torch.cat([t, pad], dim=0)

            # 프레임 수가 많은 경우 앞에서 T_max개만 사용
            elif t.shape[0] > T_max:
                t = t[:T_max]

        else:
            raise ValueError("pixel_values tensor ndim must be 3 or 4")

        proc.append(t)

    x = torch.stack(proc, dim=0)   # (B,T_max,3,H,W)
    return {"pixel_values": x, "labels": ys}
