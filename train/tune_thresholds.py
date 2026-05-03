import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["toxic", "severe_toxic", "obscene", "insult", "threat", "identity_hate"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_val(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns:
        if "comment_text" in df.columns:
            df = df.rename(columns={"comment_text": "text"})
        else:
            raise ValueError("CSV must contain 'text' or 'comment_text'")
    for c in LABELS:
        if c not in df.columns:
            raise ValueError(f"Missing label column '{c}' in {path}")
    df["text"] = df["text"].fillna("").astype(str)
    for c in LABELS:
        df[c] = df[c].fillna(0).astype(int)
    return df


@torch.inference_mode()
def predict_probs(model, tokenizer, texts: List[str], max_tokens: int, device: str, batch_size: int) -> np.ndarray:
    all_probs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(
            chunk,
            truncation=True,
            max_length=max_tokens,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.detach().float().cpu().numpy()
        all_probs.append(sigmoid(logits))

        # free VRAM aggressively
        del enc, logits
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return np.vstack(all_probs)


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray, grid: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="models/xlmr-toxic-v1")
    p.add_argument("--val_csv", default="data/processed/jigsaw2018/val.csv")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--out", default="api/thresholds.json")
    p.add_argument("--grid_steps", type=int, default=181)  # 0.05..0.95 step ~0.005
    p.add_argument("--min_t", type=float, default=0.05)
    p.add_argument("--max_t", type=float, default=0.95)
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    val_csv = Path(args.val_csv)
    out_path = Path(args.out)

    df = load_val(val_csv)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    probs = predict_probs(model, tokenizer, df["text"].tolist(), args.max_tokens, device, args.batch_size)
    y = df[LABELS].values.astype(int)

    grid = np.linspace(args.min_t, args.max_t, args.grid_steps)

    thresholds: Dict[str, float] = {}
    for i, lab in enumerate(LABELS):
        thresholds[lab] = best_threshold(y[:, i], probs[:, i], grid)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    print(f"✅ Saved thresholds to {out_path}")
    print(thresholds)


if __name__ == "__main__":
    main()