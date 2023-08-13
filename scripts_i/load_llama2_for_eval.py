import argparse
import json
from typing import Optional

from datasets import load_dataset
from llama import Llama
from tqdm import tqdm

esnli_label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_batch_size", type=int, default=4)
    parser.add_argument("--max_gen_len", type=int, default=None)
    parser.add_argument("--truncate", type=int, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    return parser.parse_args()


def init(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    return generator


def main(params):
    dialogs_template = [
        [
            {
                "role": "system",
                "content": "Predict a label for a hypothesis given a premise. The labels are entailment, neutral, and contradiction. Give your prediction following the examples below, and separate the explaination and the prediction with #### ",
            },
            {
                "role": "user",
                "content": 'Premise is "Two women are embracing while holding to go packages."\nHypothesis is "Two woman are holding packages."',
            },
            {
                "role": "assistant",
                "content": "Saying the two women are holding packages is a way to paraphrase that the packages they are holding are to go packages. #### entailment",
            },
            {
                "role": "user",
                "content": 'Premise is "Two women are embracing while holding to go packages."\nHypothesis is "The sisters are hugging goodbye while holding to go packages after just eating lunch."',
            },
            {
                "role": "assistant",
                "content": "The to go packages may not be from lunch. #### neutral",
            },
            {
                "role": "user",
                "content": 'Premise is "Two women are embracing while holding to go packages."\nHypothesis is "The men are fighting outside a deli."',
            },
            {
                "role": "assistant",
                "content": "In the first sentence there is an action of affection between women while on the second sentence there is a fight between men. #### contradiction",
            },
            {
                "role": "user",
                "content": 'Premise is "{premise}"\nHypothesis is "{hypothesis}"',
            },
        ],
    ]
    dataset_for_eval = load_dataset("esnli", split="test")
    generator = init(
        ckpt_dir=params.ckpt_dir,
        tokenizer_path=params.tokenizer_path,
        max_seq_len=params.max_seq_len,
    )

    results = []
    for i, data_dict in enumerate(tqdm(dataset_for_eval)):
        dialogs = dialogs_template.copy()
        dialogs[0][-1]["content"] = dialogs[0][-1]["content"].format(
            premise=data_dict["premise"], hypothesis=data_dict["hypothesis"]
        )
        query = dialogs[0][-1]["content"]
        label = esnli_label_map[data_dict["label"]]

        result = generator.chat_completion(
            dialogs,  # type: ignore
            temperature=params.temperature,
            top_p=params.top_p,
        )[0]
        answer = result["generation"]["content"]
        results.append({"query": query, "answer": answer, "label": label})
        if params.truncate and i + 1 == params.truncate:
            break
    with open(params.output_file, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    params = parse_args()
    main(params)
