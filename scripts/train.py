import os
import torch
import torch.nn.functional as F
import evaluate
import numpy as np

from transformers import set_seed, TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint

from src import DeepfakeModel
from src import prepare_deepfake_dataset
from src import attach_media_transforms
from src import collate_fn


def main():
    set_seed(42)
    use_bf16 = False  # 필요하면 True로 켜도 됨 (GPU가 bf16 지원해야 함)

    # ------------------------------------------------
    # 1) 데이터 로드
    # ------------------------------------------------
    train_data, test_data, label2id, id2label, class_labels, image_key, video_key = prepare_deepfake_dataset(
        data_path=None,
        data_files="/root/workspace/baseline/metadata.tsv",
        delimiter="\t",
        split="train",
        test_size=0.2,
        seed=42,
    )

    # ------------------------------------------------
    # 2) 전처리 훅 부착
    #   - 이미지: pixel_values -> (3,H,W)
    #   - 비디오: pixel_values -> (T,3,H,W)
    #   - 배치 단계(collate_fn)에서:
    #       * 이미지-only 배치  -> (B,3,H,W)
    #       * 비디오 포함 배치 -> (B,T,3,H,W)
    # ------------------------------------------------
    attach_media_transforms(
        train_data,
        test_data,
        image_key=image_key,
        video_key=video_key,
        clip_model_name="openai/clip-vit-large-patch14",
        do_face_crop=False,
        rotation_deg=15,
        num_frames=12,
    )

    # ------------------------------------------------
    # 3) 모델 생성 (CLIP backbone은 DeepfakeModel 내부에서 생성)
    # ------------------------------------------------
    num_classes = len(class_labels.names)

    model = DeepfakeModel(
        clip_model_name="openai/clip-vit-large-patch14",
        dtype="bf16" if use_bf16 else "fp32",
        freeze_backbone=True,
        unfreeze_last_n_blocks=0,
        bottleneck=256,
        num_classes=num_classes,
        hidden_mult=2,
        temporal_pool="mean",
        id2label=id2label,
        label2id=label2id,
    )

    # ------------------------------------------------
    # 4) 메트릭 (이진 전제: 0=real, 1=fake 가정)
    # ------------------------------------------------
    acc_metric = evaluate.load("accuracy")
    auc_metric = evaluate.load("roc_auc")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred  # logits: (N, num_classes)
        probs = torch.from_numpy(logits).softmax(dim=1).numpy()
        preds = probs.argmax(axis=1)

        acc = acc_metric.compute(
            predictions=preds,
            references=labels,
        )["accuracy"]

        # 이진 분류 가정: 클래스 1을 positive로 보고 AUC 계산
        if probs.shape[1] == 2:
            auc = auc_metric.compute(
                prediction_scores=probs[:, 1],
                references=labels,
                average="macro",
            )["roc_auc"]
        else:
            # num_classes > 2 인 경우 (multi-class)도 대비
            # 여기선 일단 macro-avg로 처리
            auc = auc_metric.compute(
                prediction_scores=probs,
                references=labels,
                multi_class="ovr",
                average="macro",
            )["roc_auc"]

        return {"accuracy": acc, "roc_auc": auc}

    # ------------------------------------------------
    # 5) Trainer 커스텀: 이미지/비디오 자동 분기
    # ------------------------------------------------
    class AdapterTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            x = inputs["pixel_values"]
            y = inputs["labels"].long()

            # mode 안 넘겨도 forward 안에서 dim 기준으로 image/video 자동 선택
            logits = model(pixel_values=x)

            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            return (loss, {"logits": logits}) if return_outputs else loss

    # ------------------------------------------------
    # 6) TrainingArguments
    # ------------------------------------------------
    args = TrainingArguments(
        output_dir="./outputs_multimodal",   # 비디오만이 아니라 이미지+비디오라 이름 변경
        per_device_train_batch_size=8,       # 비디오는 메모리 커서 8~16 사이에서 조절 추천
        per_device_eval_batch_size=8,
        learning_rate=1e-4,
        num_train_epochs=5,
        weight_decay=0.02,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_safetensors=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_roc_auc",
        greater_is_better=True,
        remove_unused_columns=False,         # pixel_values 유지하려면 False 필수
        dataloader_num_workers=max(4, (os.cpu_count() or 8) // 2),
        dataloader_pin_memory=True,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=50,
        report_to="none",
    )

    # ------------------------------------------------
    # 7) Trainer 생성
    # ------------------------------------------------
    trainer = AdapterTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # ------------------------------------------------
    # 8) 체크포인트 재개 여부 확인 후 학습
    # ------------------------------------------------
    last_ckpt = get_last_checkpoint(args.output_dir)
    if last_ckpt is not None:
        print(f">> resume from {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        print(">> start fresh training")
        trainer.train()


if __name__ == "__main__":
    main()
