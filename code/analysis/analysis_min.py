from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_csv(input_csv: Path, output_dir: Path):
    if not input_csv.exists():
        print("Input CSV did not exist")
        return
    if not output_dir.exists():
        print("Output dir did not exist")
        return
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()

    # Ensure correct dtypes
    bool_cols = ["judge_answer_correctness_vs_ref", "judge_answer_helpfulness"]
    for col in bool_cols:
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])

    numeric_cols = ["meta_total_tokens"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    
    # --- Clean up total_elapsed_time column ---
    def parse_seconds(x):
        """Convert strings like '12 seconds' or '1.5 s' to float seconds."""
        if pd.isna(x):
            return None
        import re
        match = re.search(r"([\d\.]+)", str(x))
        return float(match.group(1)) if match else None

    df["total_elapsed_time_sec"] = df["total_elapsed_time"].apply(parse_seconds)


    # --- ESTIMATED price per run ---
    def compute_price(row):
        """
        Compute cost (in USD) based on model type and total tokens.
        Prices per 1M tokens:
        gpt-nano: $0.40
        gpt-mini: $2.00
        """
        tokens = row.get("meta_total_tokens", 0)
        model = str(row.get("model", "")).lower()
        
        if "nano" in model:
            price_per_million = 0.40
        elif "mini" in model:
            price_per_million = 2.00
        else:
            # Default if model not recognized
            price_per_million = 1.00  

        return (tokens / 1_000_000) * price_per_million

    df["price_usd"] = df.apply(compute_price, axis=1)

    # --- Group and Aggregate ---
    group_cols = ["approach", "model", "top_k"]

    agg_df = (
        df.groupby(group_cols)
        .agg(
            avg_meta_total_tokens=("meta_total_tokens", "mean"),
            avg_price_usd=("price_usd", "mean"),
            avg_latency_sec=("total_elapsed_time_sec", "mean"),
            true_correctness_count=("judge_answer_correctness_vs_ref", "sum"),
            total_correctness_count=("judge_answer_correctness_vs_ref", "count"),
            true_helpfulness_count=("judge_answer_helpfulness", "sum"),
            total_helpfulness_count=("judge_answer_helpfulness", "count"),
        )
        .reset_index()
    )
    
    # Add percentage columns
    agg_df["pct_correctness_true"] = (
        agg_df["true_correctness_count"] / agg_df["total_correctness_count"] * 100
    )
    agg_df["pct_helpfulness_true"] = (
        agg_df["true_helpfulness_count"] / agg_df["total_helpfulness_count"] * 100
    )
    
    # --- Summary ---
    print("Summary of metrics by (approach, model, top_k):")
    print(len(agg_df))

if __name__ == "__main__":
    analyze_csv(input_csv=Path("data/localtesting/merged_output_filled_final.csv"), output_dir=Path(""))