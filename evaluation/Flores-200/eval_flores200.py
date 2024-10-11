import os
import uuid
import json
import argparse
import random
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Flores-200.")
    parser.add_argument(
        "--langs",
        nargs="+",
        help="List of dataset configurations",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        required=True,
        help="either 'eng_Latn' or 'all'",
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
        "--nshots",
        type=int,
        default=3,
        help="Number of examples to select from the dev set",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='auto',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(l) for l in f]


def write_jsonl(lst, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Writing to {path}")
    with open(path, "w", encoding="utf-8") as f:
        for item in lst:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def format_base_model_sentence_pair(src_lang, tgt_lang, src_sent, tgt_sent=None):
    if tgt_sent:
        return f"[{src_lang}]: {src_sent}\n[{tgt_lang}]: {tgt_sent}\n"
    else:
        return f"Translate the following sentence from {src_lang} to {tgt_lang}\n[{src_lang}]: {src_sent}\n[{tgt_lang}]:"


def sample_in_context_examples(
    nshots, src_sents_dev, tgt_sents_dev, src_lang, tgt_lang, seed
):
    random.seed(seed)
    indexs = [random.randint(0, len(src_sents_dev) - 1) for _ in range(nshots)]
    src_sents, tgt_sents = [], []
    for index in indexs:
        src_sents.append(src_sents_dev[index])
        tgt_sents.append(tgt_sents_dev[index])

    prefix = ""
    for src_sent, tgt_sent in zip(src_sents, tgt_sents):
        prefix += format_base_model_sentence_pair(
            src_lang, tgt_lang, src_sent, tgt_sent
        )
    return prefix


def read_sentences_from_file(path):
    with open(path, "r", encoding="utf-8") as file:
        sentences = file.readlines()
    return [sentence.strip() for sentence in sentences]


def main():
    args = parse_args()
    print(args)

    if args.model_id == "google/gemma-2-9b" or args.model_id == "google/gemma-2-9b-it":
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

    print("Loading model...")
    sampling_params = SamplingParams(
        temperature=0.6, top_p=0.9, max_tokens=2048, stop="\n"
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

    language_map = {}
    for line in args.langs:
        lang, code = line.split(" | ")
        language_map[code] = lang
    print(language_map)

    results_dir = os.path.join(args.results_dir, os.path.basename(args.model_id))
    os.makedirs(results_dir, exist_ok=True)

    dev_files = {
        code: os.path.join(args.dataset_base_path, "dev", f"{code}.dev")
        for code in language_map.keys()
    }
    devtest_files = {
        code: os.path.join(args.dataset_base_path, "devtest", f"{code}.devtest")
        for code in language_map.keys()
    }

    count = 0
    for k, v in language_map.items():
        print("==============================")
        print(f"Translating {count} / 203")
        count += 1
        src_lang = "eng_Latn" if args.tgt_lang == "all" else k
        tgt_lang = k if args.tgt_lang == "all" else "eng_Latn"
        if src_lang == tgt_lang:
            continue

        src_sents = read_sentences_from_file(devtest_files[src_lang])
        tgt_sents = read_sentences_from_file(devtest_files[tgt_lang])
        if len(src_sents) != len(tgt_sents):
            raise ValueError(
                f"Mismatched number of sentences in source and target files: {len(src_sents)} vs {len(tgt_sents)}"
            )

        src_sents_dev = read_sentences_from_file(dev_files[src_lang])
        tgt_sents_dev = read_sentences_from_file(dev_files[tgt_lang])
        if len(src_sents_dev) != len(tgt_sents_dev):
            raise ValueError(
                f"Mismatched number of sentences in source and target files: {len(src_sents)} vs {len(tgt_sents)}"
            )

        output_path = os.path.join(results_dir, f"{src_lang}-to-{tgt_lang}.jsonl")
        if os.path.exists(output_path):
            print(f"Skip {src_lang} to {tgt_lang}")
            continue

        if len(src_sents) == 0:
            print(f"No examples for {src_lang} to {tgt_lang}")
            continue
        else:
            print(
                f"Translating {len(src_sents)} examples from {src_lang} to {tgt_lang}"
            )

        random.seed(args.seed)
        seeds = [random.randint(0, 1000) for _ in range(len(src_sents))]

        results = []
        prompts = []
        for index, src_sent in enumerate(src_sents):
            if args.nshots > 0:
                prefix = sample_in_context_examples(
                    args.nshots,
                    src_sents_dev,
                    tgt_sents_dev,
                    language_map[src_lang],
                    language_map[tgt_lang],
                    seeds[index],
                )
                prompt = prefix + format_base_model_sentence_pair(
                    language_map[src_lang], language_map[tgt_lang], src_sent, None
                )
            else:
                prompt = format_base_model_sentence_pair(
                    language_map[src_lang], language_map[tgt_lang], src_sent, None
                )

            # print(prompt)
            prompts.append(prompt)

        outputs = llm.generate(prompts, sampling_params)
        for src_sent, tgt_sent, out, prompt in zip(
            src_sents, tgt_sents, outputs, prompts
        ):
            translation = out.outputs[0].text
            results.append(
                {
                    "model_name": args.model_id,
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "src_text": src_sent,
                    "ref_text": tgt_sent,
                    "hyp_text": translation,
                    "prompt": prompt,
                }
            )
        write_jsonl(results, output_path)


if __name__ == "__main__":
    print("Starting main...")
    main()
