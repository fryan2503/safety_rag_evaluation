import csv

def printFields(filename, column_name):

    count = 0
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            value = row.get(column_name, "").strip()
            if value == "":
                count += 1
                print(idx)
    print(f"Number of empty '{column_name}' fields: {count}")


printFields("./results/minimal/merged_output.csv", "judge_answer_correctness_vs_ref")
printFields("./results/minimal/merged_output.csv", "text_correctness_vs_ref")
# printFields("./results/minimal/merged_output.csv", "judge_answer_helpfulness")
# printFields("./results/minimal/merged_output.csv", "text_helpfulness")



