# EMMA-500: Enhancing Massively Multilingual Adaptation of Large Language Models


## Overview

**EMMA-500: Enhancing Massively Multilingual Adaptation** is a cutting-edge multilingual large language model designed to improve performance, particularly in low-resource languages, through continual pre-training. Built upon the Llama 2 7B architecture, **EMMA-500** leverages the **MaLA Corpus**—a diverse multilingual dataset covering over 500 languages—to push the boundaries of language modeling.

Key strengths of **EMMA-500** include enhanced commonsense reasoning, machine translation, open-ended generation, and natural language inference, making it highly effective for multilingual tasks across both high- and low-resource languages. Our carefully curated data mix ensures that the model maintains robust performance even on specialized tasks like code generation.

This repository contains the model, dataset access, benchmarks for evaluation, detailed evaluation results, and evaluation codes.

## Key Features

- **Continual Pre-training:** Extends Llama 2 7B for improved language adaptation across 546 languages.
- **MaLA Corpus:** MaLA Corpus contains over 74 billion tokens from a variety of domains.
- **Multitask Benchmarking:** Tested on a wide range of benchmarks in commonsense reasoning, machine translation, text classification, and natural language inference across low- and high-resource languages.

## Model and Dataset Access

- [EMMA-500 Model](https://huggingface.co/collections/MaLA-LM/emma-500-66eaa9acf1f512c8915b7166)
- [MaLA Corpus](https://huggingface.co/collections/MaLA-LM/mala-corpus-66e05127641a51de34d39529)
- [PolyWrite Benchmark](https://huggingface.co/datasets/MaLA-LM/PolyWrite)

## Dataset: MaLA Corpus

The **MaLA Corpus** (Massive Language Adaptation) is a multilingual dataset that facilitates continual pre-training, featuring:

- **939 languages** with over 74 billion tokens in total.
- **546 languages** containing over 100k tokens each.
- Cleaned, deduplicated versions for higher quality training.
- A wide variety of data sources, including code, books, and instruction data.

Explore more details and download the corpus on [Huggingface](https://huggingface.co/collections/MaLA-LM/mala-corpus-66e05127641a51de34d39529).

## PolyWrite Benchmark

We also introduce **PolyWrite**, a multilingual benchmark for evaluating open-ended generation tasks in 240 languages. This benchmark includes:

- **31 diverse writing tasks**, such as storytelling and email writing.
- **155 prompts** translated into multiple languages using back-translation to ensure quality.
- BLEU score filtering to maintain translation fidelity, with a total of 35,751 prompts available.

The PolyWrite dataset is accessible on [Huggingface](https://huggingface.co/datasets/MaLA-LM/PolyWrite).

## Evaluation Results

Our **EMMA-500** model was rigorously evaluated against a range of Llama 2-based models (4.5B to 13B parameters) and showed:

- **Lowest negative log-likelihood** among all models in intrinsic evaluation.
- **Significant gains** in commonsense reasoning, machine translation, and open-ended generation.
- **Outperformance** in text classification and natural language inference over all Llama 2-based models and other multilingual LLMs.
- **Improved performance** in code generation and machine reading comprehension (MRC), though some challenges remain in MRC tasks.

While **EMMA-500** performs exceptionally well in multilingual summarization and reasoning tasks, it faces challenges in low-resource languages with slightly higher Self-BLEU scores, indicating a need for further enhancements in output diversity.

## Usage

To generate text using **EMMA-500**, use the following code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "MaLA-LM/emma-500-llama2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Evaluation Codes
Evaluation codes for open-ended generation, text classification, machine translation and summarization are avialable under [./evaluation](evaluation).
For code tasks, we use a [VLLM-enabled evaluation harness package](https://github.com/iNeil77/vllm-code-harness). For other tasks, we use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).


## Citation

```
@article{ji2024emma500enhancingmassivelymultilingual,
      title={{EMMA}-500: Enhancing Massively Multilingual Adaptation of Large Language Models}, 
      author={Shaoxiong Ji and Zihao Li and Indraneil Paul and Jaakko Paavola and Peiqin Lin and Pinzhen Chen and Dayyán O'Brien and Hengyu Luo and Hinrich Schütze and Jörg Tiedemann and Barry Haddow},
      year={2024},
      journal={arXiv preprint 2409.17892},
      url={https://arxiv.org/abs/2409.17892}, 
}
```

