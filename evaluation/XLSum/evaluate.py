# This script calculates ROUGE and BERTScore metrics for text generation task XLSum.
# The script writes the evaluation metrics to a Google Sheet for easy comparison.

from rouge import Rouge
from bert_score import score as bert_score
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


def process_file(path, rouge, model_name):
    data = read_jsonl(path)
    filtered_data = [
        d
        for d in data
        if len(d["output"]) > 0 and not re.fullmatch(r"\.+", d["output"])
    ]
    refs = [[d["target"] for d in filtered_data]]
    refs4bert = [d["target"] for d in filtered_data]
    hyps = [d["output"] for d in filtered_data]
    lang_script = os.path.basename(path).split(".")[0]

    try:
        rouge_scores = rouge.get_scores(hyps, refs[0], avg=True, ignore_empty=True)
    except Exception as e:
        print(f"Error: {e}")
        print(hyps)
        return
    
    try:
        P, R, F1 = bert_score(
            hyps,
            refs4bert,
            model_type="bert-base-multilingual-cased",
            lang=None,
            rescale_with_baseline=False,
        )
        bert_score_f1 = F1.mean().item()
    except Exception as e:
        print(f"BERTScore Error: {e}")
        return

    rouge_new_row = pd.DataFrame(
        [{"Language": lang_script, model_name: rouge_scores["rouge-l"]["f"]}]
    )
    bert_score_new_row = pd.DataFrame(
        [{"Language": lang_script, model_name: bert_score_f1}]
    )
    res = {
        "lang_script": lang_script,
        "model_name": model_name,
        "rouge_score": rouge_scores["rouge-l"]["f"],
        "bert_score": bert_score_f1,
    }
    return rouge_new_row, bert_score_new_row, res


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
        rouge = Rouge()

        input_dir = os.path.join(args.input_dir, model_name)
        paths = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        # print(len(paths))
        assert len(paths) == 45
        results = []

        rouge_result_df = pd.DataFrame(columns=["Language", model_name])
        bert_score_result_df = pd.DataFrame(columns=["Language", model_name])
        for path in tqdm(paths):
            rouge_new_row, bert_score_new_row, res = process_file(path, rouge, model_name)
            rouge_result_df = pd.concat(
                [rouge_result_df, rouge_new_row], ignore_index=True
            )
            bert_score_result_df = pd.concat(
                [bert_score_result_df, bert_score_new_row], ignore_index=True
            )
            results.append(res)

        write_jsonl(results, output_file_path)
        # write2sheet(f"{args.task_name}-ROUGE", rouge_result_df)
        write2sheet(f"{args.task_name}-BERTScore", bert_score_result_df)


if __name__ == "__main__":
    main()
