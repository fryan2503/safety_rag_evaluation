import pandas as pd
import os

# === CONFIG ===
main_csv = "./results/minimal/merged_output.csv"                  # big CSV
updates_csv = "./results/minimal/merged_output_missing_filled.csv" # smaller CSV with fixed rows
output_csv = "./results/minimal/merged_output_filled.csv"          # final result

main = pd.read_csv(main_csv)
updates = pd.read_csv(updates_csv)

# === VALIDATE MERGE KEY ===
merge_key = "permutation_id"
if merge_key not in main.columns or merge_key not in updates.columns:
    raise KeyError(f"Merge key '{merge_key}' must exist in both CSVs.")

# === SET INDEX TO KEY ===
main.set_index(merge_key, inplace=True)
updates.set_index(merge_key, inplace=True)

# === REPLACE MATCHING ROWS ===
print(f"Replacing rows in main CSV where '{merge_key}' matches updates...")
before_rows = len(main)
updated_ids = list(set(main.index) & set(updates.index))
print(f"Found {len(updated_ids)} matching rows to replace.")

# This replaces the entire row for matching keys
main.update(updates)

# === RESET INDEX AND SAVE ===
main.reset_index(inplace=True)
main.to_csv(output_csv, index=False)

print(f"\nDone! Replaced {len(updated_ids)} rows.")
print(f"Saved merged CSV to: {output_csv}")
