import torch
from torch.utils.data import DataLoader

from src import prepare_deepfake_dataset
from src import attach_media_transforms
from src import get_collate_fn_stage1


def main():
    # 1. 데이터 로드 (train/test split까지)
    train_data, test_data, label2id, id2label, class_labels, image_key, video_key = \
        prepare_deepfake_dataset(
            data_path=None,
            data_files="./metadata_debug.tsv",
            delimiter="\t",
            split="train",
            test_size=0.2,
            seed=42,
        )

    # 2. transforms 부착
    #    - num_frames=1 → 비디오는 중앙/혹은 샘플된 1프레임만, 이미지는 그대로 1장
    attach_media_transforms(
        train_data,
        test_data,
        image_key=image_key,
        video_key=video_key,
        clip_model_name="openai/clip-vit-large-patch14",
        do_face_crop=False,
        rotation_deg=15,
        num_frames=1,
    )

    # 3. DataLoader (CPU 전용, collate_fn = stage1)
    collate = get_collate_fn_stage1()   # pixel_values → (B,3,H,W) 로 맞춰주는 버전
    loader = DataLoader(
        train_data,
        batch_size=4,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,   # 디버그니까 일단 0으로
    )

    # 4. 한 두 배치만 꺼내서 체크
    for i, batch in enumerate(loader):
        x = batch["pixel_values"]   # (B,3,H,W) 기대
        y = batch["labels"]         # (B,)

        print(f"[batch {i}]")
        print("  pixel_values.shape =", x.shape)
        print("  pixel_values.dtype =", x.dtype)
        print("  labels.shape       =", y.shape)
        print("  labels[:8]         =", y[:8])

        # 값 분포도 대략 확인 (정규화 되었는지)
        print("  pixel_values.min() =", float(x.min()))
        print("  pixel_values.max() =", float(x.max()))

        if i >= 2:
            break


if __name__ == "__main__":
    main()
