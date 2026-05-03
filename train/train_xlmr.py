import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import DataCollatorWithPadding
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, hamming_loss, average_precision_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

LABELS = ["toxic", "severe_toxic", "obscene", "insult", "threat", "identity_hate"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns:
        if "comment_text" in df.columns:
            df = df.rename(columns={"comment_text": "text"})
        else:
            raise ValueError("CSV must contain 'text' or 'comment_text' column.")
    for c in LABELS:
        if c not in df.columns:
            raise ValueError(f"Missing label column '{c}' in {path}")
    df["text"] = df["text"].fillna("").astype(str)
    for c in LABELS:
        df[c] = df[c].fillna(0).astype(int)
    return df


def to_hf_dataset(df: pd.DataFrame) -> Dataset:
    # HF Trainer expects labels as float tensor for multi-label BCE loss
    labels = df[LABELS].values.astype(np.float32).tolist()
    return Dataset.from_dict({"text": df["text"].tolist(), "labels": labels})


def compute_pos_weight(train_df: pd.DataFrame) -> torch.Tensor:
    # pos_weight = neg/pos for BCEWithLogitsLoss
    pos = train_df[LABELS].sum(axis=0).values.astype(np.float32)
    neg = (len(train_df) - train_df[LABELS].sum(axis=0)).values.astype(np.float32)
    pos_weight = []
    for p, n in zip(pos, neg):
        if p <= 0:
            pos_weight.append(1.0)  # avoid inf if label never appears
        else:
            pos_weight.append(float(n / p))
    return torch.tensor(pos_weight, dtype=torch.float32)


class WeightedTrainer(Trainer):
    def __init__(self, *args, pos_weight: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits

        # BCEWithLogitsLoss for multi-label
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self._pos_weight.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def build_metrics_fn() -> callable:
    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        probs = sigmoid(np.array(logits))
        y_true = np.array(labels).astype(int)

        # default threshold 0.5 for reporting during training
        y_pred = (probs >= 0.5).astype(int)

        metrics = {}
        metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        metrics["hamming_loss"] = float(hamming_loss(y_true, y_pred))

        # per-label F1
        for i, lab in enumerate(LABELS):
            metrics[f"f1_{lab}"] = float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))

        # PR-AUC (Average Precision) per label + macro
        ap_list = []
        for i, lab in enumerate(LABELS):
            try:
                ap = average_precision_score(y_true[:, i], probs[:, i])
            except ValueError:
                ap = float("nan")
            metrics[f"ap_{lab}"] = float(ap) if not np.isnan(ap) else 0.0
            ap_list.append(metrics[f"ap_{lab}"])
        metrics["ap_macro"] = float(np.mean(ap_list))

        return metrics

    return compute_metrics


def tokenize_fn(tokenizer, max_tokens: int):
    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_tokens,
            padding=False,
        )
    return tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="xlm-roberta-base")
    p.add_argument("--train_csv", default="data/processed/jigsaw2018/train.csv")
    p.add_argument("--val_csv", default="data/processed/jigsaw2018/val.csv")
    p.add_argument("--output_dir", default="models/xlmr-toxic-v1")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_csv(train_csv)
    val_df = load_csv(val_csv)

    pos_weight = compute_pos_weight(train_df)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    train_ds = to_hf_dataset(train_df).map(tokenize_fn(tokenizer, args.max_tokens), batched=True)
    val_ds = to_hf_dataset(val_df).map(tokenize_fn(tokenizer, args.max_tokens), batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
    )

    common_args = dict(
        output_dir=str(out_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        save_strategy="epoch",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=2, 
        dataloader_pin_memory=True,
    )

    # Transformers v4 uses evaluation_strategy, v5 uses eval_strategy
    try:
        train_args = TrainingArguments(**common_args, evaluation_strategy="epoch")
    except TypeError:
        train_args = TrainingArguments(**common_args, eval_strategy="epoch")
        
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = WeightedTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=build_metrics_fn(),
        pos_weight=pos_weight,
    )

    trainer.train()

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Save training metadata for reproducibility
    meta = {
        "model_name": args.model_name,
        "labels": LABELS,
        "max_tokens": args.max_tokens,
        "pos_weight": pos_weight.tolist(),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"✅ Saved checkpoint: {out_dir}")


if __name__ == "__main__":
    main()