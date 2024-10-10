# This script calculates self-BLEU scores to measure diversity in text generation task PolyWrite.
# The script writes the evaluation metrics to a Google Sheet for easy comparison.

from sacrebleu.metrics import BLEU
import json
import os
import argparse
from tqdm import tqdm
import pandas as pd
from google_sheet import write2sheet
import re


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(l) for l in f]


def write_jsonl(data, path):
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def get_self_bleu_score(bleu, corpus: list) -> float:
    # calculate self-BELU score on corpus
    score = 0.0
    cnt = 0
    length = len(corpus)

    for index in range(length):
        curr_text = corpus[index]
        other_text = corpus[:index] + corpus[index + 1 :]
        curr_belu_score = bleu.corpus_score([curr_text], [other_text])
        score += curr_belu_score.score
        cnt += 1

    return score / cnt


def process_file(path, bleu, model_name):
    data = read_jsonl(path)
    hyps = [d["output"] for d in data]
    lang_script = os.path.basename(path).split(".")[0]

    self_bleu_score = get_self_bleu_score(bleu, hyps)

    self_bleu_new_row = pd.DataFrame(
        [{"Language": lang_script, model_name: self_bleu_score}]
    )

    res = {
        "lang_script": lang_script,
        "model_name": model_name,
        "self_bleu_score": self_bleu_score,
    }
    return self_bleu_new_row, res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    args = parser.parse_args()

    model_name = args.model_id.split("/")[-1]
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = f"{output_dir}/results.jsonl"
    print(output_file_path)
    if not os.path.exists(output_file_path):
        bleu = BLEU()

        input_dir = os.path.join(args.input_dir, model_name)
        paths = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        # print(len(paths))
        assert len(paths) == 240
        results = []

        self_bleu_result_df = pd.DataFrame(columns=["Language", model_name])
        for path in tqdm(paths):
            self_bleu_new_row, res = process_file(path, bleu, model_name)
            self_bleu_result_df = pd.concat(
                [self_bleu_result_df, self_bleu_new_row], ignore_index=True
            )
            results.append(res)

        write_jsonl(results, output_file_path)
        write2sheet(f"{args.task_name}", self_bleu_result_df)


if __name__ == "__main__":
    main()
