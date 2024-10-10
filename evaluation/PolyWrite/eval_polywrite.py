import os
import json
import argparse
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PolyWrite.")
    parser.add_argument(
        "--langs",
        nargs="+",
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
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="model/path",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
    )
    return parser.parse_args()


def write_jsonl(lst, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Writing to {path}")
    with open(path, "w", encoding="utf-8") as f:
        for item in lst:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_sentences_from_file(path):
    inputs = []
    targets = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            inputs.append(data["prompt_translated"])
    return inputs


def main():
    args = parse_args()
    print(args)

    if args.model_id == "google/gemma-2-9b" or args.model_id == "google/gemma-2-9b-it":
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

    print("Loading model...")
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=512,
    )
    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        dtype=args.dtype,
        disable_custom_all_reduce=True,
        enforce_eager=True,
    )
    print("Model Loaded...")

    results_dir = os.path.join(args.results_dir, os.path.basename(args.model_id))
    os.makedirs(results_dir, exist_ok=True)

    count = 1
    for line in args.langs:
        lang, lang_script = line.split(" | ")
        print("==============================")
        print(f"Evaluating {count} / 240")
        count += 1

        inputs = read_sentences_from_file(
            os.path.join(args.dataset_base_path, f"{lang_script}.jsonl")
        )

        output_path = os.path.join(results_dir, f"{lang_script}.jsonl")
        if os.path.exists(output_path):
            print(f"Skip {lang_script}")
            continue

        results = []

        outputs = llm.generate(inputs, sampling_params)
        for input, output in zip(inputs, outputs):
            output = output.outputs[0].text
            results.append(
                {
                    "model_name": args.model_id,
                    "input": input,
                    "output": output,
                }
            )
        write_jsonl(results, output_path)


if __name__ == "__main__":
    print("Starting main...")
    main()
