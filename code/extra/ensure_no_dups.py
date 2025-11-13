import pandas as pd

if __name__ == "__main__":
    input_file = "./results/rag_generation_all_approaches_minimal.csv"

    df = pd.read_csv(input_file)

    # Find duplicates in the permutation_id column
    duplicates = df[df.duplicated(subset=["permutation_id"], keep=False)]

    if len(duplicates) > 0:
        print(f"Found {len(duplicates)} duplicate rows with the same permutation_id.")
        print(duplicates.sort_values("permutation_id"))
    else:
        print("No duplicate permutation_id values found.")
