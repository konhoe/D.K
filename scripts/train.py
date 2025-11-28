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
from src import get_collate_fn_stage1
from src import get_collate_fn_stage2


def main():
    set_seed(42)
    use_bf16 = False

    # ------------------------------------------------
    # 1) Load dataset
    # ------------------------------------------------
    train_data, test_data, label2id, id2label, class_labels, image_key, video_key = \
        prepare_deepfake_dataset(
            data_path=None,
            data_files="/root/train.tsv",
            delimiter="\t",
            split="train",
            test_size=0.2,
            seed=42,
        )

    # ------------------------------------------------
    # 2) Attach transforms (Stage-1 → 1 frame)
    # ------------------------------------------------
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

    # ------------------------------------------------
    # 3) Create model (D2ST version)
    # ------------------------------------------------
    num_classes = len(class_labels.names)

    model = DeepfakeModel(
        clip_model_name="openai/clip-vit-large-patch14",
        dtype="bf16" if use_bf16 else "fp32",
        freeze_backbone=True,
        unfreeze_last_n_blocks=0,
        num_classes=num_classes,
        d2st_num_frames=1,      # ★ Stage-1: 1 frame
        d2st_scale=0.25,        # Adapter bottleneck scale
        hidden_mult=2,
        temporal_pool="mean",
        id2label=id2label,
        label2id=label2id,
    )

    # ------------------------------------------------
    # 4) metrics
    # ------------------------------------------------
    acc_metric = evaluate.load("accuracy")
    auc_metric = evaluate.load("roc_auc")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        print(f"[compute_metrics] labels unique: {np.unique(labels)}")
        try:
            probs = torch.from_numpy(logits).softmax(dim=1).numpy()
            preds = probs.argmax(axis=1)

            acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]

            if probs.shape[1] == 2:
                auc = auc_metric.compute(
                    prediction_scores=probs[:, 1],
                    references=labels,
                    average="macro",
                )["roc_auc"]
            else:
                auc = auc_metric.compute(
                    prediction_scores=probs,
                    references=labels,
                    multi_class="ovr",
                    average="macro",
                )["roc_auc"]

            return {"accuracy": acc, "roc_auc": auc}
        except Exception as e:
            print(f"[compute_metrics] error: {e}")
            raise


    # ------------------------------------------------
    # 5) Custom Trainer
    # ------------------------------------------------
    class AdapterTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            x = inputs["pixel_values"]  # (B,3,H,W)
            y = inputs["labels"].long()

            logits = model(pixel_values=x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            return (loss, {"logits": logits}) if return_outputs else loss
        
        def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
            inputs = self._prepare_inputs(inputs)
            labels = inputs.get("labels")
            with torch.no_grad():
                logits = model(pixel_values=inputs["pixel_values"])
                loss = None
                if labels is not None:
                    loss = F.cross_entropy(logits, labels.long(), label_smoothing=0.1)
            if prediction_loss_only:
                return loss, None, None
            return loss, logits.detach(), labels.detach() if labels is not None else None


    # ------------------------------------------------
    # 6) TrainingArguments
    # ------------------------------------------------
    args = TrainingArguments(
        output_dir="./outputs_stage1",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,

        learning_rate=5e-5,
        num_train_epochs=30,
        weight_decay=0.02,
        warmup_ratio=0.1,

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_roc_auc",
        greater_is_better=True,

        remove_unused_columns=False,
        dataloader_num_workers=max(4, (os.cpu_count() or 8) // 2),
        dataloader_pin_memory=True,

        fp16=not use_bf16,
        bf16=use_bf16,
        save_safetensors=False,
        logging_steps=50,
        report_to="none",
    )

    # ------------------------------------------------
    # 7) Trainer
    # ------------------------------------------------
    trainer = AdapterTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=get_collate_fn_stage1(), 
        compute_metrics=compute_metrics,
    )


    last_ckpt = get_last_checkpoint(args.output_dir)
    if last_ckpt is not None:
        print(f">> resume from {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        print(">> start fresh training")

        trainer.train()


if __name__ == "__main__":
    main()
