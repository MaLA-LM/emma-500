# This script evaluates the performance of language models on the Aya dataset
# It calculates CHRF, BLEU, Self-BLEU, and ROUGE scores for model outputs
# The script writes the evaluation metrics to a Google Sheet for easy comparison.

from sacrebleu.metrics import CHRF, BLEU
from rouge import Rouge
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


def process_file(path, chrf, bleu, rouge, model_name):
    data = read_jsonl(path)
    filtered_data = [d for d in data if len(d["output"]) > 0 and not re.fullmatch(r'\.+', d["output"])]
    refs = [[d["target"] for d in filtered_data]]
    hyps = [d["output"] for d in filtered_data]
    lang_script = os.path.basename(path).split(".")[0]

    chrf_score = chrf.corpus_score(hyps, refs)
    bleu_score = bleu.corpus_score(hyps, refs)
    self_bleu_score = get_self_bleu_score(bleu, hyps)
    try:
        rouge_scores = rouge.get_scores(hyps, refs[0], avg=True, ignore_empty=True)
    except Exception as e:
        print(f"Error: {e}")
        print(hyps)
        return

    chrf_new_row = pd.DataFrame(
        [{"Language": lang_script, model_name: chrf_score.score}]
    )
    bleu_new_row = pd.DataFrame(
        [{"Language": lang_script, model_name: bleu_score.score}]
    )
    self_bleu_new_row = pd.DataFrame(
        [{"Language": lang_script, model_name: self_bleu_score}]
    )
    rouge_new_row = pd.DataFrame(
        [{"Language": lang_script, model_name: rouge_scores["rouge-l"]["f"]}]
    )
    res = {
        "lang_script": lang_script,
        "model_name": model_name,
        "chrf_score": chrf_score.score,
        "bleu_score": bleu_score.score,
        "self_bleu_score": self_bleu_score,
        "rouge_score": rouge_scores["rouge-l"]["f"],
    }
    return chrf_new_row, bleu_new_row, self_bleu_new_row, rouge_new_row, res


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
        chrf = CHRF(word_order=2)
        bleu = BLEU()
        rouge = Rouge()

        input_dir = os.path.join(args.input_dir, model_name)
        paths = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        # print(len(paths))
        assert len(paths) == 119
        results = []

        chrf_result_df = pd.DataFrame(columns=["Language", model_name])
        bleu_result_df = pd.DataFrame(columns=["Language", model_name])
        self_bleu_result_df = pd.DataFrame(columns=["Language", model_name])
        rouge_result_df = pd.DataFrame(columns=["Language", model_name])
        for path in tqdm(paths):
            chrf_new_row, bleu_new_row, self_bleu_new_row, rouge_new_row, res = (
                process_file(path, chrf, bleu, rouge, model_name)
            )
            chrf_result_df = pd.concat(
                [chrf_result_df, chrf_new_row], ignore_index=True
            )
            bleu_result_df = pd.concat(
                [bleu_result_df, bleu_new_row], ignore_index=True
            )
            self_bleu_result_df = pd.concat(
                [self_bleu_result_df, self_bleu_new_row], ignore_index=True
            )
            rouge_result_df = pd.concat(
                [rouge_result_df, rouge_new_row], ignore_index=True
            )
            results.append(res)

        write_jsonl(results, output_file_path)
        write2sheet(f"{args.task_name}-chrF", chrf_result_df)
        write2sheet(f"{args.task_name}-BLEU", bleu_result_df)
        write2sheet(f"{args.task_name}-Self-BLEU", self_bleu_result_df)
        write2sheet(f"{args.task_name}-ROUGE", rouge_result_df)


if __name__ == "__main__":
    main()
