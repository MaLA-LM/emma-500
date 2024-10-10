# This script evaluates machine translation quality of Flores-200 using chrF and BLEU scores.
# The script writes the evaluation metrics to a Google Sheet for easy comparison.

from sacrebleu.metrics import CHRF, BLEU
import json
import os
import argparse
from tqdm import tqdm
import pandas as pd
from google_sheet import write2sheet


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(l) for l in f]


def write_jsonl(data, path):
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def process_file(path, chrf, bleu, model_name):
    data = read_jsonl(path)
    assert len(data) == 1012
    refs = [[d["ref_text"] for d in data]]
    hyps = [d["hyp_text"] for d in data]
    chrf_score = chrf.corpus_score(hyps, refs)
    bleu_score = bleu.corpus_score(hyps, refs)
    src_lang = data[0]["src_lang"]
    tgt_lang = data[0]["tgt_lang"]

    chrf_new_row = pd.DataFrame(
        [{"Language": f"{src_lang}-{tgt_lang}", model_name: chrf_score.score}]
    )
    bleu_new_row = pd.DataFrame(
        [{"Language": f"{src_lang}-{tgt_lang}", model_name: bleu_score.score}]
    )
    res = {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "model_name": model_name,
        "chrf_score": chrf_score.score,
        "bleu_score": bleu_score.score,
    }
    return chrf_new_row, bleu_new_row, res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--translations_dir", type=str, required=True)
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
        bleu = BLEU(tokenize="flores200")

        translations_dir = os.path.join(args.translations_dir, model_name)
        paths = [
            os.path.join(translations_dir, f)
            for f in os.listdir(translations_dir)
            if os.path.isfile(os.path.join(translations_dir, f))
        ]
        assert len(paths) == 203
        results = []

        chrf_result_df = pd.DataFrame(columns=["Language", model_name])
        bleu_result_df = pd.DataFrame(columns=["Language", model_name])
        for path in tqdm(paths):
            chrf_new_row, bleu_new_row, res = process_file(path, chrf, bleu, model_name)
            chrf_result_df = pd.concat(
                [chrf_result_df, chrf_new_row], ignore_index=True
            )
            bleu_result_df = pd.concat(
                [bleu_result_df, bleu_new_row], ignore_index=True
            )
            results.append(res)

        write_jsonl(results, output_file_path)
        write2sheet(f"{args.task_name}-chrF", chrf_result_df)
        write2sheet(f"{args.task_name}-BLEU", bleu_result_df)


if __name__ == "__main__":
    main()
