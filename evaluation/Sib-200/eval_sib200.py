from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch
import random
import pandas as pd
from google_sheet import write2sheet
import numpy as np
import os
import json


def write_jsonl(lst, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Writing to {path}")
    with open(path, "w", encoding="utf-8") as f:
        for item in lst:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def generate_examples(examples):
    example_template = 'The topic of the news "{example_text}" is {example_category}'
    example_strs = [
        example_template.format(
            example_text=ex["text"], example_category=ex["category"]
        )
        for ex in examples
    ]
    return "\n".join(example_strs)


def format_text_with_examples(example, few_shot_examples):
    template = """Topic Classification: science/technology, travel, politics, sports, health, entertainment, geography.

{examples}
The topic of the news "{text}" is
"""
    examples_str = generate_examples(few_shot_examples)
    filled_template = template.format(examples=examples_str, text=example["text"])
    return filled_template


def get_ans(text, tokenizer, model, device, category_tokens):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    # inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    model.eval()
    with torch.no_grad():
        logits = (
            model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            .logits[0, -1]
            .flatten()
        )

    # Create a list of tuples having (logit, 'option') format
    options_list = [
        (logits[category_tokens["science/technology"]], "science/technology"),
        (logits[category_tokens["travel"]], "travel"),
        (logits[category_tokens["politics"]], "politics"),
        (logits[category_tokens["sports"]], "sports"),
        (logits[category_tokens["health"]], "health"),
        (logits[category_tokens["entertainment"]], "entertainment"),
        (logits[category_tokens["geography"]], "geography"),
    ]
    options_list = sorted(options_list, reverse=True)
    ans = options_list[0][1]
    return ans


def load_tsv_data(file_path):
    return pd.read_csv(file_path, delimiter="\t", index_col="index_id")


def shuffle_and_select(data, num_examples, seed):
    selected_data = data.sample(n=num_examples, random_state=seed)
    return selected_data.to_dict(orient="records")


def add_input_text(example, examples_in_train):
    return format_text_with_examples(example, examples_in_train)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate topic classification accuracy"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["zho_Hans"],
        help="List of dataset configurations",
    )
    parser.add_argument(
        "--dataset_base_path",
        type=str,
        default="dataset/path",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/path",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="model/path",
        help="Model ID for AutoModel",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the model on"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=3,
        help="Number of examples to select from the training set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, trust_remote_code=True, torch_dtype="auto", device_map="auto"
    )
    category_tokens = {
        "science/technology": tokenizer(" science/technology").input_ids[-1],
        "travel": tokenizer(" travel").input_ids[-1],
        "politics": tokenizer(" politics").input_ids[-1],
        "sports": tokenizer(" sports").input_ids[-1],
        "health": tokenizer(" health").input_ids[-1],
        "entertainment": tokenizer(" entertainment").input_ids[-1],
        "geography": tokenizer(" geography").input_ids[-1],
    }
    torch.cuda.empty_cache()

    base_path = args.dataset_base_path
    all_correct_predictions = 0
    all_total_predictions = 0
    result_df = pd.DataFrame(columns=["Language", args.model_id.split("/")[-1]])
    results_dir = os.path.join(args.results_dir, os.path.basename(args.model_id))
    os.makedirs(results_dir, exist_ok=True)

    for lang in args.configs:
        results = []
        output_path = os.path.join(results_dir, f"{lang}.jsonl")
        if os.path.exists(output_path):
            print(f"Skip {lang}")
            continue
        train_path = f"{base_path}/{lang}/train.tsv"
        test_path = f"{base_path}/{lang}/test.tsv"

        train_data = load_tsv_data(train_path)
        test_data = load_tsv_data(test_path)

        random.seed(args.seed)
        seeds = [random.randint(0, 1000) for _ in range(len(test_data))]

        test_data["input_text"] = [
            add_input_text(row, shuffle_and_select(train_data, args.num_examples, seed))
            for (row, seed) in zip(test_data.to_dict("records"), seeds)
        ]

        lang_correct_predictions = 0
        lang_total_predictions = 0

        bar = tqdm(test_data.iterrows(), total=len(test_data))
        for index, data in bar:
            predicted_category = get_ans(
                data["input_text"], tokenizer, model, args.device, category_tokens
            )
            correct_category = data["category"]
            result = {
                "model_name": args.model_id,
                "test_lang": lang,
                "prompt": data["input_text"],
                "predicted_category": predicted_category,
                "correct_category": correct_category,
            }
            results.append(result)
            if predicted_category == correct_category:
                lang_correct_predictions += 1
                all_correct_predictions += 1
            lang_total_predictions += 1
            all_total_predictions += 1

        write_jsonl(results, output_path)
        accuracy = lang_correct_predictions / lang_total_predictions
        new_row = pd.DataFrame(
            [{"Language": lang, args.model_id.split("/")[-1]: round(accuracy, 4)}]
        )
        result_df = pd.concat([result_df, new_row], ignore_index=True)
        print(f"Accuracy for language {lang}: {accuracy:.4f}")

    all_accuracy = all_correct_predictions / all_total_predictions
    new_row = pd.DataFrame(
        [
            {
                "Language": "All Languages",
                args.model_id.split("/")[-1]: round(all_accuracy, 4),
            }
        ]
    )
    result_df = pd.concat([new_row, result_df], ignore_index=True)
    print(f"Accuracy for all language: {all_accuracy:.4f}")

    write2sheet("SIB-200", result_df)


if __name__ == "__main__":
    main()
