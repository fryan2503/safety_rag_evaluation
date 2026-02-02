import pandas as pd
pd1 = pd.read_csv("./results/rag_generation_all_approaches_minimal_renamed.csv")
pd2 = pd.read_csv("./results/minimal/merged_output_filled.csv")
pd1Set = set()
pd2Set = set()
for (idx, row) in pd1.iterrows():
    pd1Set.add(row["permutation_id"])
for (idx, row) in pd2.iterrows():
    pd2Set.add(row["permutation_id"])
print(pd1Set - pd2Set)