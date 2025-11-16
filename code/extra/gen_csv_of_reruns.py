import pandas as pd

input_csv = pd.read_csv("./results/minimal/merged_output.csv")
filtered_df = input_csv[
    input_csv['judge_answer_correctness_vs_ref'].isna() | 
    input_csv['judge_answer_helpfulness'].isna()
]

filtered_df.to_csv("./results/minimal/merged_output_missing.csv", index=False)


# printFields("./results/minimal/merged_output.csv", "judge_answer_correctness_vs_ref")
# printFields("./results/minimal/merged_output.csv", "text_correctness_vs_ref")
# l1 = printFields("./results/minimal/merged_output.csv", "judge_answer_helpfulness")
# l2 = printFields("./results/minimal/merged_output.csv", "text_helpfulness")
# print(l2)


