import base64
import hashlib
import json
from typing import Any, Dict
import pandas as pd


def parse_permutation_id(pid: str, return_json: bool = False) -> Dict[str, Any]:
    """
    Decode and verify a permutation_id created by make_permutation_id().

    Returns the embedded metadata and question if integrity check passes.
    Raises ValueError if the hash does not match.
    """
    # Pad Base64 string (since padding may be stripped)
    padding = "=" * (-len(pid) % 4)
    decoded = base64.urlsafe_b64decode(pid + padding)

    # Split JSON bytes and trailing hash
    json_bytes, digest_suffix = decoded[:-8], decoded[-8:]

    # Verify hash
    expected_digest = hashlib.sha256(json_bytes).digest()[:8]
    if digest_suffix != expected_digest:
        raise ValueError("Integrity check failed — ID may be corrupted or tampered.")

    # Parse JSON payload
    if return_json:
        return json_bytes.decode("utf-8")
    payload = json.loads(json_bytes.decode("utf-8"))
    return payload

def merge_csvs(file1, file2, output_file, exclude_columns=None):
    """
    Merge two CSV files on the 'permutation_id' column, excluding specified columns.
    
    Parameters:
        file1 (str): Path to the first CSV file.
        file2 (str): Path to the second CSV file.
        output_file (str): Path where the merged CSV will be saved.
        exclude_columns (list, optional): List of column names to exclude from the final output.
    """
    # Read both CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Merge on 'permutation_id'
    merged_df = pd.merge(df1, df2, on='permutation_id', how='inner')

    # Exclude specified columns if any
    if exclude_columns:
        merged_df = merged_df.drop(columns=[col for col in exclude_columns if col in merged_df.columns])

    # Save to new CSV
    merged_df.to_csv(output_file, index=False)
    print(f"Merged CSV saved as '{output_file}'")


def process_permutation_ids(file_path, output_file):
    """
    Reads a CSV, applies parse_permutation_id() to each permutation_id,
    and saves the result in a new column.
    
    Parameters:
        file_path (str): Path to the input CSV.
        output_file (str): Path to save the new CSV.
    """
    # Load the CSV
    df = pd.read_csv(file_path)
    
    # Ensure the expected column exists
    if 'permutation_id' not in df.columns:
        raise KeyError("The CSV must contain a 'permutation_id' column.")
    
    # Apply the parse_permutation_id() function to each row
    df['parsed_permutation'] = df['permutation_id'].apply(lambda pid: parse_permutation_id(pid, True))
    
    # Save updated CSV
    df.to_csv(output_file, index=False)
    print(f"Processed CSV saved as '{output_file}'")

# Example usage:
merge_csvs('./results/minimal/batch_minimal_combined.csv', './results/rag_generation_all_approaches_minimal_renamed.csv', './results/minimal/merged_output.csv', exclude_columns=["meta_hits_text"])
# process_permutation_ids("./results/merged_output.csv", "./results/merged_output_decoded.csv")
