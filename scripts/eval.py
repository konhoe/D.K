import os
import glob
from pathlib import Path

import torch
import torch.nn.functional as F
import evaluate

from transformers import TrainingArguments, Trainer
from src import DeepfakeModel, prepare_deepfake_dataset, attach_media_transforms, collate_fn


# ===== 1. 새 validation metadata 경로 =====
VAL_METADATA = "/root/workspace/d.k./metadata_val.tsv"   # 네가 만든 val 메타데이터 경로


# ===== 2. metric 정의 (train.py에서 쓰던 거 그대로) =====
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

    # 이진 분류(0=real, 1=fake) 기준
    if probs.shape[1] == 2:
        auc = auc_metric.compute(
            prediction_scores=probs[:, 1],
            references=labels,
            average="macro",
        )["roc_auc"]
    else:
        # multi-class 대비 (혹시 모를 경우)
        auc = auc_metric.compute(
            prediction_scores=probs,
            references=labels,
            multi_class="ovr",
            average="macro",
        )["roc_auc"]

    return {"accuracy": acc, "roc_auc": auc}


# ===== 3. AdapterTrainer (train.py에서 쓰던 버전 재사용) =====
class AdapterTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        x = inputs["pixel_values"]
        y = inputs["labels"].long()

        logits = model(pixel_values=x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)

        return (loss, {"logits": logits}) if return_outputs else loss


# ===== 4. validation dataset 로더 =====
def load_val_dataset():
    # prepare_deepfake_dataset의 반환 형태에 따라 tuple unpacking은 네 코드에 맞춰 조정 필요
    # 여기서는 (val_data, _, label2id, id2label, class_labels, image_key, video_key) 가정
    val_data, _, label2id, id2label, class_labels, image_key, video_key = prepare_deepfake_dataset(
        data_files=VAL_METADATA,
        delimiter="\t",
        split="test",   # 새로운 val set만 쓸 거라서 test로 가정
        test_size=None, # split 안 쪼개고 그대로 쓰고 싶으면 이런 식으로 구현했을 가능성 있음
    )

    # eval에서는 augmentation 끄는 게 보통이라 rotation=0
    attach_media_transforms(
        val_data, val_data,
        image_key=image_key,
        video_key=video_key,
        clip_model_name="openai/clip-vit-large-patch14",
        do_face_crop=False,
        rotation_deg=0,
        num_frames=12,
    )

    num_classes = len(class_labels.names)
    return val_data, label2id, id2label, num_classes


# ===== 5. 모델 생성 함수 =====
def create_model(num_classes, label2id, id2label, use_bf16=False):
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
    return model


# ===== 6. 메인: checkpoint-* 전부 자동 평가 =====
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) val dataset 로드
    val_data, label2id, id2label, num_classes = load_val_dataset()
    print("Val samples:", len(val_data))

    # 2) 평가용 TrainingArguments (학습 X, evaluate만)
    eval_args = TrainingArguments(
        output_dir="/root/workspace/d.k./validation",
        per_device_eval_batch_size=16,
        remove_unused_columns=False,   # pixel_values 날아가면 안 됨
        dataloader_num_workers=max(4, (os.cpu_count() or 8) // 2),
        dataloader_pin_memory=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    # 3) checkpoint 리스트 가져오기
    out_dir = Path("/root/workspace/d.k./outputs_multimodal")
    ckpt_paths = sorted(
        [p for p in out_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1])  # checkpoint-1000 이런 구조 기준으로 정렬
    )

    print("Found checkpoints:", len(ckpt_paths))
    if not ckpt_paths:
        print("❌ No checkpoints found in ./outputs_multimodal")
        return

    best_auc = -1.0
    best_ckpt = None
    all_results = []

    # 4) 각 checkpoint 별로 모델 로딩 + 평가
    for ckpt in ckpt_paths:
        print(f"\n===== Evaluating {ckpt.name} =====")

        # (1) 매번 새 모델 생성
        model = create_model(num_classes, label2id, id2label, use_bf16=False)
        state_path = ckpt / "pytorch_model.bin"
        state_dict = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)

        # (2) Trainer 생성 (이 모델로만 evaluate)
        trainer = AdapterTrainer(
            model=model,
            args=eval_args,
            train_dataset=None,
            eval_dataset=val_data,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
        )

        # (3) 평가
        metrics = trainer.evaluate()
        auc = metrics.get("eval_roc_auc") or metrics.get("roc_auc")
        acc = metrics.get("eval_accuracy") or metrics.get("accuracy")

        print(f" -> AUC: {auc:.4f}, ACC: {acc:.4f}")
        all_results.append((ckpt.name, auc, acc))

        if auc is not None and auc > best_auc:
            best_auc = auc
            best_ckpt = ckpt

    # 5) 전체 요약 + best 출력
    print("\n===== Summary (New Validation Set) =====")
    for name, auc, acc in all_results:
        print(f"{name:20s} | AUC={auc:.4f} | ACC={acc:.4f}")

    print("\n>> Best checkpoint:", best_ckpt)
    print(">> Best AUC:", best_auc)


if __name__ == "__main__":
    main()
