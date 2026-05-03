import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["toxic", "severe_toxic", "obscene", "insult", "threat", "identity_hate"]

DEFAULT_RAW = Path("data/raw/jigsaw2018/train.csv")
DEFAULT_OUT_DIR = Path("data/processed/jigsaw2018")


def _load_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns:
        if "comment_text" in df.columns:
            df = df.rename(columns={"comment_text": "text"})
        else:
            raise ValueError("Missing text column. Expected 'comment_text' or 'text'.")

    missing = [c for c in LABELS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing label columns: {missing}")

    df = df[["text"] + LABELS].copy()
    df["text"] = df["text"].fillna("").astype(str)
    for c in LABELS:
        df[c] = df[c].fillna(0).astype(int)

    df["label_any"] = (df[LABELS].sum(axis=1) > 0).astype(int)
    df["label_sum"] = df[LABELS].sum(axis=1).astype(int)
    return df


def _stats(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "positives_any": int(df["label_any"].sum()),
        "positives_any_rate": float(df["label_any"].mean()),
        "per_label_positive_counts": {lab: int(df[lab].sum()) for lab in LABELS},
        "per_label_positive_rates": {lab: float(df[lab].mean()) for lab in LABELS},
    }


def _repeat_rows(df: pd.DataFrame, mask, times: int) -> pd.DataFrame:
    """Return extra copies (times-1) of rows matching mask."""
    if times <= 1:
        return df.iloc[0:0].copy()
    sub = df[mask].copy()
    if sub.empty:
        return sub
    return pd.concat([sub] * (times - 1), ignore_index=True)


def oversample_train_only(train_df: pd.DataFrame, seed: int, os_threat: int, os_severe: int, os_identity: int) -> pd.DataFrame:
    base = train_df.copy()

    add_parts = []
    if os_threat > 1:
        add_parts.append(_repeat_rows(base, base["threat"] == 1, os_threat))
    if os_severe > 1:
        add_parts.append(_repeat_rows(base, base["severe_toxic"] == 1, os_severe))
    if os_identity > 1:
        add_parts.append(_repeat_rows(base, base["identity_hate"] == 1, os_identity))

    if add_parts:
        out = pd.concat([base] + add_parts, ignore_index=True)
    else:
        out = base

    # Shuffle oversampled training set
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", default=str(DEFAULT_RAW))
    parser.add_argument("--out_dir", default="data/processed/jigsaw2018_trainonly_os")
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # Oversampling multipliers (applied ONLY to train split)
    parser.add_argument("--os_threat", type=int, default=8)
    parser.add_argument("--os_severe", type=int, default=4)
    parser.add_argument("--os_identity", type=int, default=2)

    args = parser.parse_args()

    raw_path = Path(args.raw_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = pd.read_csv(raw_path)
    df = _load_and_normalize(df)

    # ✅ Split FIRST (validation remains natural)
    train_df, val_df = train_test_split(
        df,
        test_size=args.val_size,
        random_state=args.seed,
        shuffle=True,
        stratify=df["label_any"],
    )

    # ✅ Oversample ONLY the training split
    train_os = oversample_train_only(
        train_df,
        seed=args.seed,
        os_threat=args.os_threat,
        os_severe=args.os_severe,
        os_identity=args.os_identity,
    )

    # Recompute helper columns (because we concatenated)
    for frame in (train_os, val_df):
        frame["label_any"] = (frame[LABELS].sum(axis=1) > 0).astype(int)
        frame["label_sum"] = frame[LABELS].sum(axis=1).astype(int)

    train_out = out_dir / "train.csv"
    val_out = out_dir / "val.csv"
    stats_out = out_dir / "stats.json"

    train_os.drop(columns=["label_any", "label_sum"]).to_csv(train_out, index=False)
    val_df.drop(columns=["label_any", "label_sum"]).to_csv(val_out, index=False)

    stats = {
        "source": str(raw_path),
        "val_size": args.val_size,
        "seed": args.seed,
        "labels": LABELS,
        "oversampling_train_only": True,
        "oversampling": {
            "os_threat": args.os_threat,
            "os_severe": args.os_severe,
            "os_identity": args.os_identity,
        },
        "before": {
            "train": _stats(train_df),
            "val": _stats(val_df),
        },
        "after": {
            "train_oversampled": _stats(train_os),
            "val_unchanged": _stats(val_df),
        },
    }

    stats_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"✅ Saved: {train_out}")
    print(f"✅ Saved: {val_out}")
    print(f"✅ Stats: {stats_out}")
    print("Train before:", len(train_df), "-> after oversampling:", len(train_os))
    print("Val (unchanged):", len(val_df))


if __name__ == "__main__":
    main()