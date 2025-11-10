import pandas as pd

namespace = {}
with open("./code/4_1_rag_generation.py") as f:
    exec(f.read(), namespace)

make_permutation_id = namespace["make_permutation_id"]

if __name__ == "__main__":
    input_file = "./results/rag_generation_all_approaches_minimal.csv"
    output_file = "./results/rag_generation_all_approaches_minimal_renamed.csv"

    df = pd.read_csv(input_file)
    def make_new_id(row, idx):
        permutation_source = {
            "approach": row["approach"],
            "model": row["model"],
            "top_k": row["top_k"],
            "answer_instructions_id": row["answer_instructions_id"],
            "few_shot_id": row["few_shot_id"],
            "max_tokens": row["max_tokens"],
            "reasoning_effort": row["reasoning_effort"],
            "row": idx+2
        }
        return make_permutation_id(permutation_source)

    # df["permutation_id"] = df.apply(make_new_id, axis=1)
    df["permutation_id"] = [make_new_id(row, idx) for idx, row in df.iterrows()]

    df.to_csv(output_file, index=False)
    print(f"All permutation_id values updated and written to {output_file}")
