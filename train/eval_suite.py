import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score, average_precision_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["toxic", "severe_toxic", "obscene", "insult", "threat", "identity_hate"]

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def load_multi_label_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns and "comment_text" in df.columns:
        df = df.rename(columns={"comment_text": "text"})
    if "text" not in df.columns:
        raise ValueError(f"{path} missing 'text' column")
    for c in LABELS:
        if c not in df.columns:
            raise ValueError(f"{path} missing label '{c}'")
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
        logits = model(**enc).logits.detach().cpu().numpy()
        all_probs.append(sigmoid(logits))
    return np.vstack(all_probs)

def eval_multilabel(y_true: np.ndarray, probs: np.ndarray, thresholds: Dict[str, float]) -> Dict[str, float]:
    thr = np.array([thresholds[l] for l in LABELS], dtype=np.float32)
    y_pred = (probs >= thr).astype(int)

    out = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }
    for i, lab in enumerate(LABELS):
        out[f"f1_{lab}"] = float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
        try:
            out[f"ap_{lab}"] = float(average_precision_score(y_true[:, i], probs[:, i]))
        except ValueError:
            out[f"ap_{lab}"] = 0.0
    return out

def load_thresholds(path: Path) -> Dict[str, float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    for lab in LABELS:
        if lab not in d:
            d[lab] = 0.5
    return {k: float(d[k]) for k in LABELS}

def eval_multilingual_toxic_only(multilingual_csv: Path, probs_toxic: np.ndarray) -> Dict[str, float]:
    """
    Multilingual dataset is typically single-label toxicity, not 6-label.
    We evaluate only the 'toxic' dimension.
    """
    df = pd.read_csv(multilingual_csv)
    if "comment_text" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"comment_text": "text"})
    if "text" not in df.columns:
        raise ValueError(f"{multilingual_csv} missing text column")
    if "toxic" not in df.columns:
        raise ValueError(f"{multilingual_csv} missing 'toxic' column (needed for toxic-only eval)")
    y = df["toxic"].astype(int).values

    out = {}
    try:
        out["toxic_roc_auc"] = float(roc_auc_score(y, probs_toxic))
    except ValueError:
        out["toxic_roc_auc"] = 0.0
    try:
        out["toxic_ap"] = float(average_precision_score(y, probs_toxic))
    except ValueError:
        out["toxic_ap"] = 0.0

    pred = (probs_toxic >= 0.5).astype(int)
    out["toxic_f1@0.5"] = float(f1_score(y, pred, zero_division=0))
    return out

def unintended_bias_metrics(df: pd.DataFrame, toxic_probs: np.ndarray, identities: List[str],
                           min_count: int = 50, identity_threshold: float = 0.5) -> Dict[str, float]:
    """
    Simplified bias metrics for Unintended Bias dataset (toxicity-only).

    Detects ground truth column automatically among:
      target, toxicity, toxic, label
    """
    gt_col = None
    for cand in ["target", "toxicity", "toxic", "label"]:
        if cand in df.columns:
            gt_col = cand
            break

    if gt_col is None:
        return {
            "note": "Bias eval skipped: no ground-truth column found (expected target/toxicity/toxic/label).",
            "columns_head": list(df.columns[:60]),
        }

    gt = df[gt_col].fillna(0.0).values
    y = (gt >= 0.5).astype(int)

    out: Dict[str, object] = {"ground_truth_column": gt_col}

    try:
        out["overall_auc"] = float(roc_auc_score(y, toxic_probs))
    except ValueError:
        out["overall_auc"] = 0.0

    def safe_auc(y_true, y_score):
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_score))

    per_id = {}
    for ident in identities:
        if ident not in df.columns:
            continue
        subgroup = (df[ident].fillna(0.0).values >= identity_threshold)
        if subgroup.sum() < min_count:
            continue

        auc_sub = safe_auc(y[subgroup], toxic_probs[subgroup])

        bpsn_mask = ((y == 1) & (~subgroup)) | ((y == 0) & (subgroup))
        auc_bpsn = safe_auc(y[bpsn_mask], toxic_probs[bpsn_mask])

        bnsp_mask = ((y == 0) & (~subgroup)) | ((y == 1) & (subgroup))
        auc_bnsp = safe_auc(y[bnsp_mask], toxic_probs[bnsp_mask])

        per_id[ident] = {
            "subgroup_count": int(subgroup.sum()),
            "subgroup_auc": auc_sub if auc_sub is not None else 0.0,
            "bpsn_auc": auc_bpsn if auc_bpsn is not None else 0.0,
            "bnsp_auc": auc_bnsp if auc_bnsp is not None else 0.0,
        }

    out["identities_evaluated"] = len(per_id)
    if per_id:
        out["mean_subgroup_auc"] = float(np.mean([v["subgroup_auc"] for v in per_id.values()]))
        out["mean_bpsn_auc"] = float(np.mean([v["bpsn_auc"] for v in per_id.values()]))
        out["mean_bnsp_auc"] = float(np.mean([v["bnsp_auc"] for v in per_id.values()]))
    else:
        out["mean_subgroup_auc"] = 0.0
        out["mean_bpsn_auc"] = 0.0
        out["mean_bnsp_auc"] = 0.0

    out["per_identity"] = per_id
    return out


def load_unintended_bias_subset(
    csv_path: Path,
    bias_read_rows: int,
    bias_max_rows: int,
    bias_seed: int,
    bias_split: str | None,
    identity_cols: List[str],
) -> pd.DataFrame:
    """
    Practical loader for very large Unintended Bias CSV:
    - reads only needed columns (usecols)
    - reads only first N rows if bias_read_rows > 0
    - optional filter by split column (e.g. 'test')
    - samples down to bias_max_rows
    """
    gt_candidates = ["target", "toxicity", "toxic", "label"]
    usecols = ["comment_text", "split"] + gt_candidates + identity_cols
    header = pd.read_csv(csv_path, nrows=0)
    existing = [c for c in usecols if c in header.columns]
    read_kwargs = dict(usecols=existing, low_memory=False)
    if bias_read_rows and bias_read_rows > 0:
        read_kwargs["nrows"] = bias_read_rows

    df = pd.read_csv(csv_path, **read_kwargs)
    if bias_split and "split" in df.columns:
        df = df[df["split"].astype(str) == bias_split].copy()

    if bias_max_rows and bias_max_rows > 0 and len(df) > bias_max_rows:
        df = df.sample(n=bias_max_rows, random_state=bias_seed).reset_index(drop=True)

    if "comment_text" not in df.columns:
        raise ValueError("Bias CSV missing 'comment_text' column.")

    df["comment_text"] = df["comment_text"].fillna("").astype(str)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="models/xlmr-toxic-v1")
    p.add_argument("--thresholds", default="api/thresholds.json")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--jigsaw_val", default="data/processed/jigsaw2018/val.csv")
    p.add_argument("--multilingual_validation", default="data/raw/jigsaw2020_multi/validation.csv")
    p.add_argument("--unintended_bias_all", default="data/raw/jigsaw2019_bias/all_data.csv")
    p.add_argument("--bias_max_rows", type=int, default=50000,
                   help="Max rows used for bias evaluation after filtering (0 = no cap).")
    p.add_argument("--bias_read_rows", type=int, default=300000,
                   help="Read only first N rows from the huge CSV for speed (0 = read full file).")
    p.add_argument("--bias_seed", type=int, default=42)
    p.add_argument("--bias_split", default="test",
                   help="If CSV has 'split' column, use this split (e.g. 'test' or 'train'). Empty to disable.")
    p.add_argument("--report_out", default="reports/metrics_remote.json")
    args = p.parse_args()
    model_dir = Path(args.model_dir)
    thresholds_path = Path(args.thresholds)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    thresholds = load_thresholds(thresholds_path)
    report = {
        "model_dir": str(model_dir),
        "thresholds": thresholds,
        "max_tokens": args.max_tokens,
        "device": device,
        "batch_size": args.batch_size,
    }

    jigsaw_val = load_multi_label_csv(Path(args.jigsaw_val))
    probs = predict_probs(model, tokenizer, jigsaw_val["text"].tolist(), args.max_tokens, device, args.batch_size)
    y = jigsaw_val[LABELS].values.astype(int)
    report["jigsaw2018_val"] = eval_multilabel(y, probs, thresholds)
    mpath = Path(args.multilingual_validation)
    if mpath.exists():
        mdf = pd.read_csv(mpath)
        if "comment_text" in mdf.columns and "text" not in mdf.columns:
            mdf = mdf.rename(columns={"comment_text": "text"})
        if "text" in mdf.columns:
            mprobs = predict_probs(model, tokenizer, mdf["text"].fillna("").astype(str).tolist(),
                                   args.max_tokens, device, args.batch_size)
            toxic_probs = mprobs[:, 0]
            if "toxic" in mdf.columns:
                report["multilingual_toxic_only"] = eval_multilingual_toxic_only(mpath, toxic_probs)
            else:
                report["multilingual_toxic_only"] = {"note": "validation.csv has no 'toxic' label column, skipped."}
        else:
            report["multilingual_toxic_only"] = {"note": "validation.csv missing text/comment_text column, skipped."}
    else:
        report["multilingual_toxic_only"] = {"note": "multilingual validation.csv not found, skipped."}

    bias_path = Path(args.unintended_bias_all)
    if bias_path.exists():
        identity_cols = [
            "male", "female", "homosexual_gay_or_lesbian", "christian", "jewish", "muslim",
            "black", "white", "psychiatric_or_mental_illness",
        ]
        bias_split = args.bias_split.strip() if isinstance(args.bias_split, str) else args.bias_split
        if bias_split == "":
            bias_split = None

        bdf = load_unintended_bias_subset(
            csv_path=bias_path,
            bias_read_rows=args.bias_read_rows,
            bias_max_rows=args.bias_max_rows,
            bias_seed=args.bias_seed,
            bias_split=bias_split,
            identity_cols=identity_cols,
        )

        report["unintended_bias_sample"] = {
            "bias_read_rows": args.bias_read_rows,
            "bias_max_rows": args.bias_max_rows,
            "bias_split": bias_split,
            "rows_used": int(len(bdf)),
            "seed": args.bias_seed,
        }

        texts = bdf["comment_text"].tolist()
        bprobs = predict_probs(model, tokenizer, texts, args.max_tokens, device, args.batch_size)
        toxic_probs = bprobs[:, 0]
        report["unintended_bias_toxic_only"] = unintended_bias_metrics(
            bdf, toxic_probs, identity_cols, min_count=50, identity_threshold=0.5
        )
    else:
        report["unintended_bias_toxic_only"] = {"note": "bias CSV not found, skipped."}

    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report: {out}")

if __name__ == "__main__":
    main()