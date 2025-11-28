# scripts/smoke_cpu.py
from __future__ import annotations
import os
import torch
from torch.utils.data import DataLoader

from transformers import set_seed

from src import (
    DeepfakeModel,
    prepare_deepfake_dataset,
    attach_media_transforms,
    get_collate_fn_stage1,
    get_collate_fn_stage2
)

def main():
    set_seed(42)

    print("✅ CPU 스모크 테스트 시작")

    # ------------------------------------------------
    # 1) 데이터 로드 (너가 방금 쓰던 metadata 경로 그대로)
    #    → 일단 200개만 써서 빠르게 확인
    # ------------------------------------------------
    train_data, test_data, label2id, id2label, class_labels, image_key, video_key = (
        prepare_deepfake_dataset(
            data_path=None,
            data_files="/Users/junyoung/Desktop/baseline/metadata_1000.tsv",  # 네 tsv 경로
            delimiter="\t",
            split="train",
            test_size=0.2,
            seed=42,
        )
    )

    # 너무 많으면 느리니까 앞에서 몇 개만 잘라서 확인
    train_small = train_data.select(range(min(32, len(train_data))))

    print(f"🔹 train_small size = {len(train_small)}")
    print(f"🔹 image_key = {image_key}, video_key = {video_key}")

    # ------------------------------------------------
    # 2) 전처리 훅 부착 (이미지/비디오 → pixel_values 텐서)
    #    지금은 num_frames=1 로 이미지 스테이지만 확인
    # ------------------------------------------------
    attach_media_transforms(
        train_small,
        test_data,  # 안 써도 되지만 형식상 넘겨줌
        image_key=image_key,
        video_key=video_key,
        clip_model_name="openai/clip-vit-large-patch14",
        do_face_crop=False,
        rotation_deg=15,
        num_frames=12,  # CPU 테스트니까 1장만
    )

    # ------------------------------------------------
    # 3) DataLoader (Stage-1용 collate_fn: (B,3,H,W) 나오는 버전)
    # ------------------------------------------------
    loader = DataLoader(
        train_small,
        batch_size=4,
        shuffle=True,
        num_workers=0,              
        collate_fn=get_collate_fn_stage2(),  
    )

    # ------------------------------------------------
    # 4) 모델 생성 (CPU, fp32, frame=1 기준 cfg)
    # ------------------------------------------------
    device = torch.device("cpu")
    num_classes = len(class_labels.names)

    model = DeepfakeModel(
        clip_model_name="openai/clip-vit-large-patch14",
        dtype="fp32",              # CPU에서는 fp32가 제일 안전
        freeze_backbone=True,      # 일단 backbone 동결
        unfreeze_last_n_blocks=0,
        num_classes=num_classes,
        d2st_num_frames=12,         # 지금은 이미지 한 장만 → T=1
        d2st_scale=0.25,
        hidden_mult=2,
        temporal_pool="mean",
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    model.eval()

    # ------------------------------------------------
    # 5) 한두 배치만 forward 돌려보기
    # ------------------------------------------------
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch["pixel_values"].to(device)  # (B,3,H,W)
            y = batch["labels"].to(device)        # (B,)

            print(f"\n[batch {i}]")
            print(f"  pixel_values.shape = {x.shape}")
            print(f"  labels.shape       = {y.shape}")
            print(f"  labels[:8]         = {y[:8]}")

            # ✨ 여기서 실제 모델 forward
            logits = model(pixel_values=x)        # mode=None → dim=4라 image branch
            probs = torch.softmax(logits, dim=-1)

            print(f"  logits.shape       = {logits.shape}")  # (B, num_classes)
            print(f"  probs[0]           = {probs[0]}")

            # 한두 배치만 보면 되니까
            if i >= 2:
                break

    print("\n✅ CPU 스모크 테스트 완료!")


if __name__ == "__main__":
    main()
