from datasets import load_dataset
import pandas as pd
import jsonlines
import os

dataset_human_annotated = load_dataset(
    "CohereForAI/aya_evaluation_suite", "aya_human_annotated"
)
dataset_machine_translated = load_dataset(
    "CohereForAI/aya_evaluation_suite", "dolly_machine_translated"
)

df_human_annotated = pd.DataFrame(dataset_human_annotated["test"])
# print(df_human_annotated.columns)
df_machine_translated = pd.DataFrame(dataset_machine_translated["test"])
# print(df_machine_translated.columns)


df = pd.concat([df_human_annotated, df_machine_translated], ignore_index=True)
print(df.columns)
df["language_script"] = df["language"] + "_" + df["script"]

df = df[["id", "inputs", "targets", "language_script"]]

for group_name, group_df in df.groupby("language_script"):
    path = f"/your/data/path/{group_name}.jsonl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with jsonlines.open(path, mode="a") as writer:
        for _, row in group_df.iterrows():
            writer.write({"inputs": row["inputs"], "targets": row["targets"]})
