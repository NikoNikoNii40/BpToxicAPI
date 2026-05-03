from pathlib import Path
import pandas as pd

RAW = Path("data/raw/jigsaw2019_bias/all_data.csv")
OUT = Path("data/processed/jigsaw2019_bias/eval.csv")

# Minimal identity columns for quick bias checks (you can expand later)
IDENTITY_COLS = [
    "male", "female", "homosexual_gay_or_lesbian", "christian", "jewish", "muslim",
    "black", "white", "psychiatric_or_mental_illness",
]

def main():
    if not RAW.exists():
        print(f"[SKIP] Unintended Bias all_data.csv not found: {RAW}")
        print("Place the Kaggle file here (recommended):")
        print("  data/raw/jigsaw2019_bias/all_data.csv")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW)

    # Normalize column name to 'text' if needed
    if "text" not in df.columns and "comment_text" in df.columns:
        df = df.rename(columns={"comment_text": "text"})

    keep = ["text", "target"] + [c for c in IDENTITY_COLS if c in df.columns]
    df = df[keep].copy()
    df["text"] = df["text"].fillna("").astype(str)

    df.to_csv(OUT, index=False)
    print(f"✅ Saved bias eval file: {OUT} (rows={len(df)})")

if __name__ == "__main__":
    main()