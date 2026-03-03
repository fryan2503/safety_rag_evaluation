import sys
import pandas as pd

DUP_COLUMNS = [
    "question",
    "gold_answer",
    "top_k",
    "reasoning_effort",
    "model",
    "approach",
]

def main(csv_path: str):
    df = pd.read_csv(csv_path)

    missing = [c for c in DUP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Mark duplicates (keep=False flags all members of duplicate groups)
    dup_mask = df.duplicated(subset=DUP_COLUMNS, keep=False)

    dup_df = df[dup_mask].sort_values(DUP_COLUMNS)

    if dup_df.empty:
        print("No duplicates found.")
        return

    print(f"Found {len(dup_df)} duplicate rows")
    print("\nDuplicate groups:\n")

    # Group and display
    for keys, group in dup_df.groupby(DUP_COLUMNS):
        print("—" * 80)
        for col, val in zip(DUP_COLUMNS, keys):
            print(f"{col}: {val}")
        print(f"Rows: {len(group)}")
        print(group.index.tolist())

    # Optional: save duplicates to file
    out = csv_path.replace(".csv", "_duplicates.csv")
    dup_df.to_csv(out, index=False)
    print(f"\nDuplicate rows written to: {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_duplicates.py <results.csv>")
        sys.exit(1)

    main(sys.argv[1])
