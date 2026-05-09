from pathlib import Path
import pandas as pd

RAW = Path("data/raw/jigsaw2020_multi/validation.csv")
OUT = Path("data/processed/jigsaw2020_multi/eval.csv")

def main():
    if not RAW.exists():
        print(f"[SKIP] Multilingual validation not found: {RAW}")
        print("Place the Kaggle file here (recommended):")
        print("  data/raw/jigsaw2020_multi/validation.csv")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW)

    if "text" not in df.columns and "comment_text" in df.columns:
        df = df.rename(columns={"comment_text": "text"})

    keep = [c for c in ["text", "toxic"] if c in df.columns]
    df = df[keep].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df.to_csv(OUT, index=False)
    print(f"Saved multilingual eval file: {OUT} (rows={len(df)})")

if __name__ == "__main__":
    main()